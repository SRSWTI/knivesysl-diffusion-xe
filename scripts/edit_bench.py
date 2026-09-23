#!/usr/bin/env python3
"""Runnable local editing evaluation, isolated checks, and Unix-socket UI service."""
import argparse
import fcntl
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler
import json
import os
import re
from pathlib import Path
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'scripts' / 'edit_cases'
STATE = ROOT / '.cache' / 'edit-checks'
SOCKET = STATE / 'checks.sock'
MANIFEST = {case['id']: case for case in json.loads((CASES / 'cases.json').read_text())}
CHECK_LOCK = threading.Lock()
CUDA_ROOT = Path(shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc').resolve().parent.parent


def public_cases():
    result = []
    for case in MANIFEST.values():
        item = {key: value for key, value in case.items() if key not in ('reference', 'harness')}
        item['code'] = (CASES / case['file']).read_text()
        item['lines'] = len(item['code'].splitlines())
        item['toolchain'] = {'cpp': 'g++ C++20 + undefined-behavior sanitizer', 'cuda': 'nvcc sm_120 + real NVIDIA GPU', 'python': 'Python + supplied dataset tests'}[case['language']]
        result.append(item)
    return result


def sandbox_command(work, command, *, compile_phase=False, cuda=False):
    if not shutil.which('bwrap'):
        raise RuntimeError('bubblewrap is required; refusing to execute generated code without isolation')
    args = ['bwrap', '--die-with-parent', '--new-session', '--unshare-all', '--cap-drop', 'ALL',
            '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib', '--ro-bind', '/lib64', '/lib64',
            '--symlink', 'usr/bin', '/bin', '--symlink', 'usr/sbin', '/sbin',
            '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--dir', '/etc',
            '--clearenv', '--setenv', 'PATH', f'{CUDA_ROOT}/bin:/usr/bin:/bin',
            '--setenv', 'HOME', '/tmp', '--setenv', 'TMPDIR', '/tmp', '--setenv', 'LANG', 'C.UTF-8',
            '--setenv', 'LD_LIBRARY_PATH', f'{CUDA_ROOT}/lib64:/usr/lib/x86_64-linux-gnu',
            '--bind' if compile_phase else '--ro-bind', str(work), '/work', '--chdir', '/work']
    if Path('/etc/ld.so.cache').exists():
        args += ['--ro-bind', '/etc/ld.so.cache', '/etc/ld.so.cache']
    if cuda and not compile_phase:
        args += ['--ro-bind', '/sys', '/sys']
        for device in sorted(Path('/dev').glob('nvidia*')):
            args += ['--dev-bind', str(device), str(device)]
    limits = ['/usr/bin/prlimit', '--core=0', '--cpu=60' if compile_phase else '--cpu=40',
              '--fsize=134217728' if compile_phase else '--fsize=1048576']
    # CUDA reserves a large virtual address range; an address-space cap prevents
    # driver initialization. Limit its data segment instead, and keep buffers tiny.
    limits += ['--data=2147483648'] if cuda and not compile_phase else ['--as=4294967296' if compile_phase else '--as=1073741824']
    return args + limits + ['--'] + command


def execute(work, command, *, compile_phase=False, cuda=False):
    started = time.perf_counter()
    output_file = work / ('compile.log' if compile_phase else 'run.log')
    with output_file.open('wb') as output:
        process = subprocess.Popen(sandbox_command(work, command, compile_phase=compile_phase, cuda=cuda),
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=90 if compile_phase else 60)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    with output_file.open('r', errors='replace') as output:
        text = output.read(16000)
    return {'returncode': process.returncode, 'timeout': timed_out, 'output': text,
            'seconds': round(time.perf_counter()-started, 4)}


def check_code(case_id, code):
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _check_code(case_id, code)


def _check_code(case_id, code):
    if not isinstance(case_id, str) or case_id not in MANIFEST:
        raise ValueError('Unknown evaluation case')
    if not isinstance(code, str) or len(code) > 150000:
        raise ValueError('Source must be a string of at most 150000 characters')
    case = MANIFEST[case_id]
    language = case['language']
    STATE.mkdir(parents=True, exist_ok=True)
    result = {'case': case_id, 'language': language, 'source_sha256': hashlib.sha256(code.encode()).hexdigest()}
    with tempfile.TemporaryDirectory(prefix='job-', dir=STATE) as temporary:
        work = Path(temporary)
        suffix = {'python': '.py', 'cpp': '.hpp', 'cuda': '.cuh'}[language]
        (work / ('subject'+suffix)).write_text(code)
        if language == 'python':
            records = json.loads((CASES / 'leetcode_records.json').read_text())
            record = records[case['dataset_task']]
            (work / 'tests.json').write_text(json.dumps({'task_id':case['dataset_task'], 'entry_point':record['entry_point'], 'tests':record['tests']}))
            shutil.copyfile(CASES / 'check_dataset.py', work / 'check.py')
            compile_command = ['/usr/bin/python3', '-I', '-m', 'py_compile', '/work/subject.py']
            run_command = ['/usr/bin/python3', '-I', '/work/check.py']
        elif language == 'cpp':
            shutil.copyfile(CASES / case['harness'], work / 'check.cpp')
            compile_command = ['/usr/bin/g++', '-std=c++20', '-O2', '-Wall', '-Wextra', '-fsanitize=undefined',
                               '-fno-sanitize-recover=all', '/work/check.cpp', '-o', '/work/check']
            run_command = ['/work/check']
        else:
            shutil.copyfile(CASES / case['harness'], work / 'check.cu')
            compile_command = [str(CUDA_ROOT / 'bin/nvcc'), '-std=c++17', '-O2', '-arch=sm_120', '/work/check.cu', '-o', '/work/check']
            run_command = ['/work/check']
        result['compile'] = execute(work, compile_command, compile_phase=True, cuda=language=='cuda')
        if result['compile']['returncode'] != 0:
            result['status'] = 'compile_timeout' if result['compile']['timeout'] else 'compile_error'
            return result
        result['run'] = execute(work, run_command, cuda=language=='cuda')
        completed = False
        if language == 'python':
            try:
                result['tests'] = json.loads(result['run']['output'])
                tests = result['tests']
                completed = (isinstance(tests, dict) and tests.get('total') == len(record['tests'])
                             and tests.get('passed') == len(record['tests']))
            except json.JSONDecodeError:
                pass
        else:
            completed = re.search(r'(?m)^PASS [1-9][0-9]* .* checks$', result['run']['output']) is not None
        result['status'] = ('run_timeout' if result['run']['timeout'] else
                            'test_failure' if result['run']['returncode'] != 0 else
                            'pass' if completed else 'invalid_test_result')
        return result


class UnixHTTPConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(SOCKET))


def socket_request(method, path, payload=None):
    connection = UnixHTTPConnection('localhost', timeout=150)
    try:
        connection.request(method, path, body=json.dumps(payload) if payload is not None else None,
                           headers={'Content-Type':'application/json'})
        response = connection.getresponse()
        body = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(str(body))
        return body
    finally:
        connection.close()


class CheckServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(format % args, flush=True)

    def reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/health':
            self.reply(200, {'ready':True,'busy':CHECK_LOCK.locked(),'pid':os.getpid()})
        elif self.path == '/examples':
            self.reply(200, {'cases':public_cases()})
        else:
            self.reply(404, {'error':'Not found'})

    def do_POST(self):
        if self.path == '/shutdown':
            self.reply(200, {'stopping':True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if self.path != '/check':
            self.reply(404, {'error':'Not found'})
            return
        length = int(self.headers.get('Content-Length', '0'))
        if not 0 < length <= 1000000:
            self.reply(413, {'error':'Invalid request size'})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('Request must be an object')
        except (ValueError, TypeError) as error:
            self.reply(400, {'error':str(error)})
            return
        if not CHECK_LOCK.acquire(blocking=False):
            self.reply(409, {'error':'Another compile/run check is in progress'})
            return
        try:
            result = check_code(payload.get('case'), payload.get('code'))
            self.reply(200, result)
        except ValueError as error:
            self.reply(400, {'error':str(error)})
        except Exception as error:
            self.reply(500, {'error':str(error)})
        finally:
            CHECK_LOCK.release()


def serve():
    STATE.mkdir(parents=True, exist_ok=True)
    if SOCKET.exists():
        try:
            socket_request('GET','/health')
        except (OSError, RuntimeError):
            SOCKET.unlink()
        else:
            raise RuntimeError('Check service is already running')
    with CheckServer(str(SOCKET), Handler) as server:
        os.chmod(SOCKET, 0o660)
        print(f'Checks listening on Unix socket {SOCKET}', flush=True)
        try:
            server.serve_forever()
        finally:
            SOCKET.unlink(missing_ok=True)


def start_service():
    try:
        print(json.dumps(socket_request('GET','/health')))
        return
    except (OSError, RuntimeError):
        pass
    for binary in ('bwrap','g++','nvcc'):
        if not shutil.which(binary):
            raise RuntimeError(f'{binary} is required for the configured evaluation cases')
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'service.log').open('ab') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'serve'],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    for _ in range(100):
        if child.poll() is not None:
            raise RuntimeError(f'Check service failed; see {STATE / "service.log"}')
        try:
            print(json.dumps(socket_request('GET','/health')))
            return
        except OSError:
            time.sleep(.1)
    raise RuntimeError('Timed out starting check service')


def evaluate(base, output, selected=None):
    output.mkdir(parents=True, exist_ok=True)
    report = {'model': json.load(urllib.request.urlopen(base+'/v1/models')), 'started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'attempts_per_condition':1, 'dataset':'newfacade/LeetCodeDataset', 'cases':[], 'controls':[]}
    chosen = [case for case in MANIFEST.values() if not selected or case['id'] in selected]
    for case in chosen:
        original = (CASES / case['file']).read_text()
        baseline = check_code(case['id'], original)
        reference = check_code(case['id'], case['reference'])
        control = {'case':case['id'],'original':baseline,'reference':reference}
        report['controls'].append(control)
        (output/'report.json').write_text(json.dumps(report,indent=2))
        print(f"CONTROL {case['id']}: original={baseline['status']} reference={reference['status']}",flush=True)
        if reference['status'] != 'pass':
            raise RuntimeError(f"Reference checker failed for {case['id']}; refusing to score model output")
        if baseline['status'] == 'pass':
            raise RuntimeError(f"Original fixture unexpectedly passes for {case['id']}")
        for mode in ('selection','whole'):
            for specificity in ('vague','detailed'):
                name = f"{case['id']}--{mode}--{specificity}"
                directory = output/name
                directory.mkdir()
                payload = {'code':original,'instruction':case['prompts'][specificity], 'language':case['language'], 'mode':mode, 'max_tokens':8192}
                if mode == 'selection':
                    payload.update(selection_start=case['selection_start'],selection_end=case['selection_end'])
                (directory/'request.json').write_text(json.dumps(payload,indent=2))
                result = {'case':case['id'],'mode':mode,'specificity':specificity,'original_lines':len(original.splitlines()),'directory':str(directory)}
                started = time.perf_counter()
                terminal = None
                final = None
                preserved = True
                try:
                    request = urllib.request.Request(base+'/edit',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(request,timeout=600) as response, (directory/'events.jsonl').open('w') as log:
                        for line in response:
                            if not line.startswith(b'data: '): continue
                            event = json.loads(line[6:])
                            log.write(json.dumps(event)+'\n')
                            if mode == 'selection' and 'code' in event:
                                prefix = original[:case['selection_start']]
                                suffix = original[case['selection_end']:]
                                preserved &= event['code'].startswith(prefix) and event['code'].endswith(suffix) and len(event['code'])>=len(prefix)+len(suffix)
                            if event['type'] in ('done','error'): terminal=event
                            if event['type']=='done': final=event['code']
                    result.update(seconds=round(time.perf_counter()-started,3), terminal=terminal, outside_preserved=preserved)
                    if final is None:
                        result['status']='generation_error'
                    elif not preserved:
                        result['status']='boundary_failure'
                    else:
                        (directory/case['file']).write_text(final)
                        result['check']=check_code(case['id'],final)
                        result['status']=result['check']['status']
                        import difflib
                        result['changed_lines']=sum(line.startswith(('+','-')) and not line.startswith(('+++','---')) for line in difflib.unified_diff(original.splitlines(),final.splitlines()))
                except Exception as error:
                    result.update(status='request_error',error=str(error),seconds=round(time.perf_counter()-started,3))
                if result.get('terminal'):
                    result['terminal']={key:value for key,value in result['terminal'].items() if key!='code'}
                report['cases'].append(result)
                (output/'report.json').write_text(json.dumps(report,indent=2))
                print(f"{name}: {result['status']} ({result['seconds']}s)",flush=True)
    report['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['start','serve','stop','run','check','controls'])
    parser.add_argument('--base',default='http://127.0.0.1:8080')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--case',action='append')
    parser.add_argument('--source',type=Path)
    args=parser.parse_args()
    if args.command=='start': start_service()
    elif args.command=='serve': serve()
    elif args.command=='stop':
        try: print(json.dumps(socket_request('POST','/shutdown')))
        except OSError: print('Check service is not running')
    elif args.command=='check':
        if not args.case or len(args.case)!=1 or not args.source: parser.error('check requires one --case and --source')
        print(json.dumps(check_code(args.case[0],args.source.read_text()),indent=2))
    elif args.command=='controls':
        for case in MANIFEST.values():
            if args.case and case['id'] not in args.case: continue
            print(json.dumps({'case':case['id'],'original':check_code(case['id'],(CASES/case['file']).read_text()),'reference':check_code(case['id'],case['reference'])},indent=2),flush=True)
    else:
        output=args.output or ROOT/'artifacts'/('editing-eval-'+time.strftime('%Y%m%d-%H%M%S'))
        evaluate(args.base.rstrip('/'),output,args.case)
        print(f'Report: {output / "report.json"}')


if __name__=='__main__':
    main()

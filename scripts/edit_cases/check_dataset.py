#!/usr/bin/env python3
"""Run unmodified dataset expectations plus separately labeled semantic checks."""
import copy
import json
from pathlib import Path
import runpy
import sys
import time


def main():
    data = json.loads(Path('/work/tests.json').read_text())
    namespace = runpy.run_path('/work/subject.py', run_name='candidate')
    method = data['entry_point'].split('.',1)[1]
    candidate = getattr(namespace['Solution'](), method)
    passed = 0
    failures = []
    semantic_passed = 0
    semantic_total = 0
    out_of_contract = 0
    mutated_inputs = 0
    contract_passed = 0
    contract_total = 0
    started = time.perf_counter()
    for index, case in enumerate(data['tests']):
        arguments = copy.deepcopy(case['kwargs'])
        result = None
        error = None
        try:
            result = candidate(**arguments)
            matched = result == case['expected']
        except Exception as caught:
            error = type(caught).__name__ + ': ' + str(caught)
            matched = False
        mutated_inputs += arguments != case['kwargs']
        passed += bool(matched)
        if not matched and len(failures) < 8:
            failures.append({'test':index,'input':repr(case['kwargs'])[:300],
                             'expected':repr(case['expected'])[:300], 'actual':repr(result)[:300],'error':error})
        if data['task_id'] == 'two-sum':
            nums = case['kwargs']['nums']
            target = case['kwargs']['target']
            seen = set()
            exists = False
            for value in nums:
                if target-value in seen: exists=True; break
                seen.add(value)
            if not exists:
                out_of_contract += 1
            else:
                semantic_total += 1
                valid = (error is None and isinstance(result,(list,tuple)) and len(result)==2 and
                         all(type(i) is int and 0<=i<len(nums) for i in result) and
                         result[0]!=result[1] and nums[result[0]]+nums[result[1]]==target)
                semantic_passed += bool(valid)
        if data['task_id'] == 'coin-change':
            values = case['kwargs']
            in_contract = (0 <= values['amount'] <= 10000 and 1 <= len(values['coins']) <= 12
                           and all(1 <= coin <= 2**31-1 for coin in values['coins']))
            if in_contract:
                contract_total += 1
                contract_passed += bool(matched)
    report = {'passed':passed,'total':len(data['tests']),'failures':failures,'mutated_input_cases':mutated_inputs,
              'seconds':round(time.perf_counter()-started,4),'grading':'exact dataset input_output expectations'}
    if data['task_id']=='two-sum':
        report['valid_pair_semantics']={'passed':semantic_passed,'total':semantic_total,
                                        'out_of_contract_no_solution_cases':out_of_contract,
                                        'note':'Supplementary only; raw dataset failures are retained. Any valid index order/pair is accepted here.'}
    if data['task_id'] == 'coin-change':
        report['published_contract'] = {'passed':contract_passed,'total':contract_total,
                                        'out_of_contract_cases':len(data['tests'])-contract_total,
                                        'note':'Supplementary only; every raw dataset case was still executed and graded.'}
    print(json.dumps(report))
    return 0 if passed==len(data['tests']) else 1


if __name__=='__main__':
    sys.exit(main())

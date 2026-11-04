"""Exact complete-candidate bounds without storing an exponential assignment list."""
from math import prod

def bounds(cv_correct, query_correct, query_n):
    assert len(cv_correct)==len(query_correct)>0 and query_n>0
    eligible=[[j for j,v in enumerate(row) if v==max(row)] for row in cv_correct]
    assert all(len(row)==3 for row in cv_correct+query_correct)
    lo=sum(min(q[j] for j in choices) for q,choices in zip(query_correct,eligible))
    hi=sum(max(q[j] for j in choices) for q,choices in zip(query_correct,eligible))
    oracle=sum(max(q) for q in query_correct)
    scale=100/(len(query_correct)*query_n)
    return {'eligible':eligible,'assignment_count':prod(map(len,eligible)),
            'mean_accuracy_bounds':[lo*scale,hi*scale],
            'mean_regret_bounds':[(oracle-hi)*scale,(oracle-lo)*scale],
            'frequency_bounds':{j:[sum(choices==[j] for choices in eligible),sum(j in choices for choices in eligible)] for j in range(3)}}

if __name__=='__main__':
    import itertools,json
    cv=[[5,5,4],[2,3,3]];q=[[10,20,90],[90,30,40]]
    b=bounds(cv,q,100)
    values=[sum(q[i][j] for i,j in enumerate(assignment))*100/200 for assignment in itertools.product(*b['eligible'])]
    assert b['mean_accuracy_bounds']==[min(values),max(values)] and b['assignment_count']==4
    large=bounds([[4,4,4]]*12,[[20,50,70]]*12,100)
    assert large['assignment_count']==3**12 and large['mean_accuracy_bounds']==[20,70]
    assert large['mean_regret_bounds']==[0,50]
    print(json.dumps({'status':'PASS','small_enumeration_matches':True,'all_531441_assignments_covered_analytically':True,'no_candidate_truncation':True}))

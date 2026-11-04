"""Use integer correct-count sums for strict scientific inequalities."""
def accounting(full, selected_indices, cv_correct, query_n):
    n=len(full);oracle=sum(max(r) for r in full)
    best_constant=max(sum(row[p] for row in full) for p in range(3))
    selected=sum(row[p] for row,p in zip(full,selected_indices))
    best_tied=sum(max(row[p] for p,c in enumerate(cv) if c==max(cv)) for row,cv in zip(full,cv_correct))
    H=oracle-best_constant;C=oracle-selected;Cmin=oracle-best_tied
    scale=100/(n*query_n)
    return {'oracle_correct_sum':oracle,'best_constant_correct_sum':best_constant,'selected_correct_sum':selected,
            'best_cv_tie_correct_sum':best_tied,'H_correct_sum':H,'C_correct_sum':C,'Cmin_correct_sum':Cmin,
            'H_pp':H*scale,'C_pp':C*scale,'C_gt_H':C>H,'Cmin_gt_H':Cmin>H}

if __name__=='__main__':
    a=accounting([[81,83,82],[83,84,85]],[1,1],[[1,2,0],[1,2,0]],100)
    assert a['C_correct_sum']==a['H_correct_sum'] and not a['C_gt_H']
    assert not a['Cmin_gt_H']
    b=accounting([[81,83,82],[83,84,85]],[0,0],[[2,2,0],[2,2,0]],100)
    assert b['C_gt_H'] and not b['Cmin_gt_H']
    print('PASS: exact equality and favorable-tie equality are not strict excess')

import cPickle as pkl

dst = pkl.load(open('../NBA_SPORTSVU_SAMPLE_FILE/pd_distance.pkl', 'rb'))
ddt_dst = pkl.load(open('../NBA_SPORTSVU_SAMPLE_FILE/pd_ddt_distance.pkl', 'rb'))
dirc = pkl.load(open('../NBA_SPORTSVU_SAMPLE_FILE/pd_direction.pkl', 'rb'))

idxs = list(dst.index)
candidate_idxs = []
candidate2_idxs = []

for idx in idxs:
    # check index consistency
    print 'current idx = ', idx
    if not ( (idx in dst.index) and (idx in ddt_dst.index) and (idx in dirc.index) ) :
        continue

    # check size of data
    if (dst.ix[idx, :].shape[0] != 17) or (ddt_dst.ix[idx, :].shape[0] != 17) or (dirc.ix[idx, :].shape[0] != 17):
        continue

    # check player's ID consistency
    dst_idx = dst.ix[idx, :].dropna()
    ddt_dst_idx = ddt_dst.ix[idx, :].dropna()
    dirc_idx = dirc.ix[idx, :].dropna()
    ID_dst = dst_idx.index
    ID_ddt_dst = ddt_dst_idx.index
    ID_dirc = dirc_idx.index
    if not ((len(set(ID_dst).intersection(set(ID_ddt_dst))) == 10) and
            (len(set(ID_ddt_dst).intersection(set(ID_dirc))) == 10) and
            (len(set(ID_dst).intersection(set(ID_dirc))) == 10)):
        continue

    # check the 3 condition matches or not
    idx_cnt = 0
    for playID in ID_dst:
        if ((dst_idx.ix[playID] < 10.0) and
            (ddt_dst_idx.ix[playID] < -10.0) and
            (dirc_idx.ix[playID] < 10.0)):
            print playID
            print idx
            print dst_idx
            print ddt_dst_idx
            print dirc_idx
            idx_cnt += 1

            # past 1sec data obtainable?
            if dst_idx.ix[(idx+1:idx]

            candidate_idxs.append(('%.2f' % idx))

            
            raw_input()
    if idx_cnt >= 2:
        candidate2_idxs.append(('%.2f' % idx))
    idx_cnt = 0
    
#    print "the following is okay data!"
#    print idx  # playerID list
#    raw_input()
#        if dst.columns == (idx in ddt_dst.index) and (idx in dirc.index):
print 'hoge'
print candidate_idxs
print candidate2_idxs

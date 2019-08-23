import sys
import pickle
import argparse
import re
import MySQLdb
import rouge

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from myutils import prep, drop, statusout, batch_gen, seq2sent, index2word

def prepare_results(p, r, f, metric):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

def average_scores(scores):
    if len(scores)==0: return None
    output = {}
    for rouge_type in scores[0]:
        output[rouge_type] = {}
        for metric in scores[0][rouge_type]:
            output[rouge_type][metric] = mean([score[rouge_type][metric] for score in scores if score])
    return output

def rouge_score(hyps, refs, ne):
    #hyps = [hyps[i] for i in range(len(hyps)) if len(refs[i])>0]
    #refs = [refs[i] for i in range(len(refs)) if len(refs[i])>0]
    #if ne:
    #    refs = [refs[i] for i in range(len(refs)) if len(hyps[i])>0]
    #    hyps = [hyps[i] for i in range(len(hyps)) if len(hyps[i])>0]

    #if len(hyps) == 0: return None
    
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                   max_n=4,
                   limit_length=True,
                   length_limit=100,
                   length_limit_type='words',
                   alpha=.5, # Default F1_score
                   weight_factor=1.2)

    scores = evaluator.get_scores(hyps, refs)
    return scores

def get_rouge_score(hyps, refs, ne):
    #mult_hyps = not isinstance(hyps[0], str)
    #if mult_hyps:
    #    scores = []
    #    for i in range(len(refs)):
    #        ref_scores = [rouge_score([hyps[i][user]], [refs[i]], ne) for user in hyps[i]]
    #        scores.append(average_scores([score for score in ref_scores if score != None]))
    #    scores = average_scores([score for score in scores if score != None])
    #else:
    scores = rouge_score(hyps, refs, ne)
    return scores

def get_indiv_rouge_score(hyps, refs, ne):
    users = set([user for hyp in hyps for user in hyp])
    hyps = {user: [labels[user] if user in labels else None for labels in hyps] for user in users}
    all_scores = {}
    for user in hyps:
        user_refs = [refs[i] for i in range(len(refs)) if hyps[user][i] != None]
        user_hyps = [hyps[user][i] for i in range(len(hyps[user])) if hyps[user][i] != None]
        user_score = rouge_score(user_hyps, user_refs, ne)
        if user_score: all_scores[user] = user_score
    rouge_scores = [score for score in all_scores.values() if score]
    rouge_max = {rougetype: {metric: max([rouge_scores[i][rougetype][metric] for i in range(len(rouge_scores))])for metric in rouge_scores[0][rougetype]} for rougetype in rouge_scores[0]}
    rouge_min = {rougetype: {metric: min([rouge_scores[i][rougetype][metric] for i in range(len(rouge_scores))])for metric in rouge_scores[0][rougetype]} for rougetype in rouge_scores[0]}
    rouge_mean = {rougetype: {metric: mean([rouge_scores[i][rougetype][metric] for i in range(len(rouge_scores))])for metric in rouge_scores[0][rougetype]} for rougetype in rouge_scores[0]}
    rouge_median = {rougetype: {metric: median([rouge_scores[i][rougetype][metric] for i in range(len(rouge_scores))])for metric in rouge_scores[0][rougetype]} for rougetype in rouge_scores[0]}
    rouge_stddev = {rougetype: {metric: stdev([rouge_scores[i][rougetype][metric] for i in range(len(rouge_scores))])for metric in rouge_scores[0][rougetype]} for rougetype in rouge_scores[0]}
    return(rouge_max, rouge_min, rouge_mean, rouge_median, rouge_stddev)

def print_scores(scores):
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        print(prepare_results(results['p'], results['r'], results['f'], metric))

def print_indiv_scores(scores):
    print("Max:")
    print_scores(scores[0])
    print("Min:")
    print_scores(scores[1])
    print("Mean:")
    print_scores(scores[2])
    print("Median:")
    print_scores(scores[3])
    print("Stdev:")
    print_scores(scores[4])

def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret

def re_0002(i):
    # split camel case and remove special characters
    tmp = i.group(0)
    if len(tmp) > 1:
        if tmp.startswith(' '):
            return tmp
        else:
            return '{} {}'.format(tmp[0], tmp[1])
    else:
        return ' '.format(tmp)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str, default=None)
    parser.add_argument('--data', dest='dataprep', type=str, default='/nfs/projects/attn-to-fc/data/standard/output')  
    parser.add_argument('--outdir', dest='outdir', type=str, default='/nfs/projects/attn-to-fc/data/outdir')
    parser.add_argument('--challenge', action='store_true', default=False)
    parser.add_argument('--obfuscate', action='store_true', default=False)
    parser.add_argument('--sbt', action='store_true', default=False)
    args = parser.parse_args()
    outdir = args.outdir
    dataprep = args.dataprep
    input_file = args.input
    challenge = args.challenge
    obfuscate = args.obfuscate
    sbt = args.sbt

    if challenge:
        dataprep = '../data/challengeset/output'

    if obfuscate:
        dataprep = '../data/obfuscation/output'

    if sbt:
        dataprep = '../data/sbt/output'

    if input_file is None:
        print('Please provide an input file to test with --input')
        exit()

    sys.path.append(dataprep)
    import tokenizer

    prep('preparing predictions list... ')
    preds = dict()
    predicts = open(input_file, 'r')
    for c, line in enumerate(predicts):
        (fid, pred) = line.split('\t')
        fid = int(fid)
        pred = pred.split()
        pred = fil(pred)
        preds[fid] = ' '.join(pred)
    predicts.close()
    drop()

    #db = MySQLdb.connect(host='localhost', user='ports_20k', passwd='s3m3rU', db='sourcerer')
    #cur = db.cursor()
    re_0001_ = re.compile(r'([^a-zA-Z0-9 ])|([a-z0-9_][A-Z])')

    refs = list()
    newpreds = list()
    d = 0
    targets = open('%s/coms.test' % (dataprep), 'r')
    for line in targets:
        (fid, com) = line.split(',')
        fid = int(fid)
        com = com.split()
        com = ' '.join(fil(com))
        
        if len(com) < 1:
            continue

        #q = 'select name from functionalunits where id='+str(fid)
        #cur.execute(q)
        #for tname in cur.fetchall():
        #    fname = re_0001_.sub(re_0002, str(tname))
        #    fname = fname.lower()
        #    fname = fname.rstrip()
        #    fname = fname.lstrip()

        #print(fname, com)

        #c = 0
        #for word in fname.split(' '):
        #    if word in com:
        #        c += 1
        #print(fname, com, c, len(fname.split(' ')))
        #if (c / len(fname.split(' '))) >= 0.5:
        #    continue

        try:
            newpreds.append(preds[fid])
        except Exception as ex:
            #newpreds.append([])
            continue
        
        refs.append(com)

    print('final status')
    print_scores(get_rouge_score(newpreds, refs, False))
    #print(rouge_so_far(evaluator, refs, newpreds))


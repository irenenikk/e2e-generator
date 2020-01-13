id=$1
echo 'Estimating bleu score using testset'
python -u generator/generate.py rest_e2e/testset_w_refs.csv -id "$1" -b 10 > "bleu_score_det_testset_beam_$id.txt"
echo 'First estimation done'
python -u generator/generate.py rest_e2e/testset_w_refs.csv -id "$1" -b 0 > "bleu_score_det_testset_token_sampled_$id.txt"
echo 'Scond estimation done'
python -u generator/generate.py rest_e2e/testset_w_refs.csv -s -cpd cpd_model_mmhc.pkl -id "$1" -b 10 > "bleu_score_content_sampled_testset_beam_$id.txt"
echo 'Third estimation done'
python -u generator/generate.py rest_e2e/testset_w_refs.csv -s -cpd cpd_model_mmhc.pkl -id "$1" -b 0 > "bleu_score_content_sampled_testset_token_sampled_$id.txt"
echo 'Fourth estimation done'

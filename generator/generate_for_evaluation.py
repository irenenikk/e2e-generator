from data_manager import load_text_data
from generate import generate_reference_using_beam, generate_reference_with_sampling
import argparse
import helpers
from models import Encoder, BahdanauAttention, Decoder
import tensorflow as tf

parser = argparse.ArgumentParser(description='Generate a file of own utterances and reference utterances in order ot ')
parser.add_argument("test_data", type=str,
                    help="The path to the test data file")
parser.add_argument("-id", "--identifier", default='',
                    help="The identifier used to define checkpoint and training info directories")
parser.add_argument("-b", "--beam-width", type=int, default=0,
                    help="Size of beam to use in generation. If not specified use sampling.")
parser.add_argument("-s", "--sample-content", default=False, action="store_true",
                    help="Sample slots used in MR of utterance.")
parser.add_argument("-cpd", "--cpd-model-file", default='cpd_model_mmhc.pkl',
                    help="Pickle file where the cpd model is stored")

if __name__ == "__main__":
    # restoring the latest checkpoint in checkpoint_dir
    args = parser.parse_args()
    test_data_file = args.test_data
    identifier = args.identifier
    checkpoint_dir = 'training_checkpoints' + identifier
    training_info_file = 'training_info' + identifier + '.pkl'
    references_filename = 'references' + identifier + '.txt'
    model_output_filename = 'model_output' + identifier + '.txt'
    beam_width = args.beam_width
    sample_content = args.sample_content
    print('Sampling content', sample_content)
    cpd_model_file = args.cpd_model_file
    training_info = helpers.load_from_pickle(training_info_file)
    encoder = Encoder(len(training_info['mr_word2idx'])+1, 
                        training_info['embedding_dim'], 
                        training_info['units'])
    decoder = Decoder(len(training_info['ref_word2idx'])+1, 
                        training_info['embedding_dim'], 
                        training_info['units']*2, 
                        training=False)
    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)
    print('Restoring checkpoint from', checkpoint_dir)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    test_data = load_text_data(test_data_file)
    prev_mr_info = None
    references_file = open(references_filename,"w+")
    model_output_file = open(model_output_filename,"w+")
    for i in range(len(test_data)):
        mr_input = test_data['mr'].iloc[i]
        reference = test_data['ref'].iloc[i]
        references_file.write(reference + '\n')
        if mr_input != prev_mr_info:
            # write new line to the model file
            mr_info = ''
            if sample_content:
                mr_info = sample_mr_content(mr_input, cpd_model_file)
            else:
                mr_info = mr_input
            generated = ''
            if beam_width > 0:
                generated = generate_reference_using_beam(encoder, decoder, mr_info, training_info, beam_width)
            else:
                generated = generate_reference_with_sampling(encoder, decoder, mr_info, training_info)
            model_output_file.write(generated + '\n')
            references_file.write('\n')
        prev_mr_info = mr_input

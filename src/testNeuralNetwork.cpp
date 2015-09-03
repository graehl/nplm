#include <tclap/CmdLine.h>
#include <boost/algorithm/string/join.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>

#include "model.h"
#include "propagator.h"
#include "neuralClasses.h"
#include "param.h"
#include "util.h"

using namespace std;
using namespace boost;
using namespace TCLAP;
using namespace Eigen;

using namespace nplm;

int main (int argc, char *argv[]) 
{
    param myParam;

    try {
      // program options //
      CmdLine cmd("Tests a two-layer neural probabilistic language model.", ' ' , "0.1");

      ValueArg<int> debug("", "debug", "Debug level. Higher debug levels print log-probabilities of each n-gram (level 1), and n-gram itself (level 2). Default: 0.", false, 0, "int", cmd);

      ValueArg<int> num_threads("", "num_threads", "Number of threads. Default: maximum.", false, 0, "int", cmd);
      SwitchArg premultiply("", "premultiply", "premultiply hidden layer.", cmd, false);
      SwitchArg unnormalized("", "unnormalized", "do not normalize output.", cmd, false);
      ValueArg<int> minibatch_size("", "minibatch_size", "Minibatch size. Default: 64.", false, 64, "int", cmd);

      ValueArg<string> arg_test_file("", "test_file", "Test file (one numberized example per line).", true, "", "string", cmd);

      ValueArg<string> arg_model_file("", "model_file", "Model file.", true, "", "string", cmd);

      cmd.parse(argc, argv);

      myParam.model_file = arg_model_file.getValue();
      myParam.test_file = arg_test_file.getValue();

      myParam.num_threads  = num_threads.getValue();
      myParam.premultiply  = premultiply.getValue();
      myParam.normalization  = !unnormalized.getValue();
      myParam.minibatch_size = minibatch_size.getValue();
      myParam.debug = debug.getValue();

      cerr << "Command line: \n";
      cerr << boost::algorithm::join(vector<string>(argv, argv+argc), " ") << '\n';
      
      const string sep(" Value: ");
      cerr << arg_model_file.getDescription() << sep << arg_model_file.getValue() << '\n';
      cerr << arg_test_file.getDescription() << sep << arg_test_file.getValue() << '\n';
    
      cerr << num_threads.getDescription() << sep << num_threads.getValue() << '\n';
    }
    catch (TCLAP::ArgException &e)
    {
      cerr << "error: " << e.error() <<  " for arg " << e.argId() << '\n';
      exit(1);
    }

    myParam.num_threads = setup_threads(myParam.num_threads);

    ///// Create network and propagator

    model nn;
    nn.read(myParam.model_file);
    myParam.ngram_size = nn.ngram_size;
    propagator prop(nn, myParam.minibatch_size);

    ///// Set param values according to what was read in from model file

    myParam.ngram_size = nn.ngram_size;
    myParam.input_vocab_size = nn.input_vocab_size;
    myParam.output_vocab_size = nn.output_vocab_size;
    myParam.num_hidden = nn.num_hidden;
    myParam.input_embedding_dimension = nn.input_embedding_dimension;
    myParam.output_embedding_dimension = nn.output_embedding_dimension;

    if (myParam.premultiply) {
      cerr << "Premultiplying hidden layer\n";
      nn.premultiply();
    }

    ///// Read test data

    vector<int> test_data_flat;
    readDataFile(myParam.test_file, myParam.ngram_size, test_data_flat);
    int test_data_size = test_data_flat.size() / myParam.ngram_size;
    cerr << "Number of test instances: " << test_data_size << '\n';

    Map< Matrix<int,Dynamic,Dynamic> > test_data(test_data_flat.data(), myParam.ngram_size, test_data_size);
    
    ///// Score test data

    int num_batches = (test_data_size-1)/myParam.minibatch_size + 1;
    cerr<<"Number of test minibatches: "<<num_batches<<'\n';

    double log_likelihood = 0.0;
    
    Matrix<double,Dynamic,Dynamic> scores(nn.output_vocab_size, myParam.minibatch_size);
    Matrix<double,Dynamic,Dynamic> output_probs(nn.output_vocab_size, myParam.minibatch_size);

    for (int batch = 0; batch < num_batches; batch++)
    {
	int minibatch_start_index = myParam.minibatch_size * batch;
	int current_minibatch_size = min(myParam.minibatch_size,
					 test_data_size - minibatch_start_index);
	Matrix<int,Dynamic,Dynamic> minibatch = test_data.middleCols(minibatch_start_index, current_minibatch_size);
	
	prop.fProp(minibatch.topRows(myParam.ngram_size-1));

	if (myParam.normalization)
	{
	    // Do full forward prop through output word embedding layer
	    if (prop.skip_hidden)
		prop.output_layer_node.param->fProp(prop.first_hidden_activation_node.fProp_matrix, scores);
	    else
		prop.output_layer_node.param->fProp(prop.second_hidden_activation_node.fProp_matrix, scores);


	    // And softmax and loss
	    double minibatch_log_likelihood;
	    SoftmaxLogLoss().fProp(scores.leftCols(current_minibatch_size), 
				  minibatch.row(myParam.ngram_size-1), 
				  output_probs,
				  minibatch_log_likelihood);
	    log_likelihood += minibatch_log_likelihood;
	}
	else
	{
	    for (int j=0; j<current_minibatch_size; j++)
	    {
	        int output = minibatch(nn.ngram_size-1, j);
                if (prop.skip_hidden)
                    output_probs(output, j) = prop.output_layer_node.param->fProp(prop.first_hidden_activation_node.fProp_matrix, output, j);
                else
                    output_probs(output, j) = prop.output_layer_node.param->fProp(prop.second_hidden_activation_node.fProp_matrix, output, j);
		log_likelihood += output_probs(output, j);
	    }
	}

        if (myParam.debug > 0) {
          for (int i=0; i<current_minibatch_size; i++) {
            if (myParam.debug > 1) {
              cerr << minibatch.block(0,i,myParam.ngram_size,1).transpose() << " ";
            }
            cerr << output_probs(minibatch(myParam.ngram_size-1,i),i) << '\n';
          }
        }
	
    }	
    cerr << "Test log-likelihood: " << log_likelihood << '\n';
}

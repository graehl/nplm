#include <cstdlib>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <cctype>

#include "model.h"
#include "param.h"

using namespace std;
using namespace boost;
using namespace boost::random;

namespace nplm
{

/**
   \return maximum iter on [begin, end) with pred(*iter)==true, else return end. this is like find_if
   (c.rbegin(), c.rend(), pred)+1).base() except you get end rather than begin if not found.
*/
template <class Iter, class Pred>
inline Iter findLast(Iter begin, Iter const& end, Pred pred) {
  Iter i = end;
  while (i > begin) {
    --i;
    if (pred(*i)) return i;
  }
  return end;
}

struct NotSpace {
  typedef bool result_type;
  bool operator()(char c) const {
#ifdef _WIN32
    return (c & 0x80) || !std::isspace(c);
#else
    return !std::isspace(c);
#endif
  }
};

static inline std::string& rightTrim(std::string& s) {
  std::string::iterator end = s.end();
  std::string::iterator i = findLast(s.begin(), s.end(), NotSpace());
  if (i == end)
    s.clear();
  else
    s.erase(++i, end);
  return s;
}

void model::resize(int ngram_size,
                   int input_vocab_size,
                   int output_vocab_size,
                   int input_embedding_dimension,
                   int num_hidden,
                   int output_embedding_dimension)
{
  input_layer.resize(input_vocab_size, input_embedding_dimension, ngram_size-1);
  if (num_hidden == 0)
  {
    first_hidden_linear.resize(output_embedding_dimension, input_embedding_dimension*(ngram_size-1));
    first_hidden_activation.resize(output_embedding_dimension);
    second_hidden_linear.resize(1,1);
    second_hidden_activation.resize(1);
  }
  else
  {
    first_hidden_linear.resize(num_hidden, input_embedding_dimension*(ngram_size-1));
    first_hidden_activation.resize(num_hidden);
    second_hidden_linear.resize(output_embedding_dimension, num_hidden);
    second_hidden_activation.resize(output_embedding_dimension);
  }
  output_layer.resize(output_vocab_size, output_embedding_dimension);
  this->ngram_size = ngram_size;
  this->input_vocab_size = input_vocab_size;
  this->output_vocab_size = output_vocab_size;
  this->input_embedding_dimension = input_embedding_dimension;
  this->num_hidden = num_hidden;
  this->output_embedding_dimension = output_embedding_dimension;
  premultiplied = false;
}

void model::initialize(boost::random::mt19937 &init_engine,
                       bool init_normal,
                       double init_range,
                       double init_bias,
                       string &parameter_update,
                       double adagrad_epsilon)
{
  input_layer.initialize(init_engine,
                         init_normal,
                         init_range,
                         parameter_update,
                         adagrad_epsilon);
  output_layer.initialize(init_engine,
                          init_normal,
                          init_range,
                          init_bias,
                          parameter_update,
                          adagrad_epsilon);
  first_hidden_linear.initialize(init_engine,
                                 init_normal,
                                 init_range,
                                 parameter_update,
                                 adagrad_epsilon);
  second_hidden_linear.initialize(init_engine,
                                  init_normal,
                                  init_range,
                                  parameter_update,
                                  adagrad_epsilon);
}

void model::premultiply()
{
  // Since input and first_hidden_linear are both linear,
  // we can multiply them into a single linear layer *if* we are not training
  int context_size = ngram_size-1;
  Matrix<double,Dynamic,Dynamic> U = first_hidden_linear.U;
  if (num_hidden == 0)
  {
    first_hidden_linear.U.resize(output_embedding_dimension, input_vocab_size * context_size);
  }
  else
  {
    first_hidden_linear.U.resize(num_hidden, input_vocab_size * context_size);
  }
  for (int i=0; i<context_size; i++)
    first_hidden_linear.U.middleCols(i*input_vocab_size, input_vocab_size) = U.middleCols(i*input_embedding_dimension, input_embedding_dimension) * input_layer.W->transpose();
  input_layer.W->resize(1,1); // try to save some memory
  premultiplied = true;
}

void model::readConfig(istream &config_file)
{
  string line;
  vector<string> fields;
  int ngram_size, vocab_size, input_embedding_dimension, num_hidden, output_embedding_dimension;
  activation_function_type activation_function = this->activation_function;
  while (getline(config_file, line) && line != "")
  {
    splitBySpace(line, fields);
    if (fields[0] == "ngram_size")
      ngram_size = lexical_cast<int>(fields[1]);
    else if (fields[0] == "vocab_size")
      input_vocab_size = output_vocab_size = lexical_cast<int>(fields[1]);
    else if (fields[0] == "input_vocab_size")
      input_vocab_size = lexical_cast<int>(fields[1]);
    else if (fields[0] == "output_vocab_size")
      output_vocab_size = lexical_cast<int>(fields[1]);
    else if (fields[0] == "input_embedding_dimension")
      input_embedding_dimension = lexical_cast<int>(fields[1]);
    else if (fields[0] == "num_hidden")
      num_hidden = lexical_cast<int>(fields[1]);
    else if (fields[0] == "output_embedding_dimension")
      output_embedding_dimension = lexical_cast<int>(fields[1]);
    else if (fields[0] == "activation_function")
      activation_function = string_to_activation_function(fields[1]);
    else if (fields[0] == "version")
    {
      int version = lexical_cast<int>(fields[1]);
      if (version != 1)
      {
        cerr << "error: file format mismatch (expected 1, found " << version << ")\n";
        exit(1);
      }
    }
    else
      cerr << "warning: unrecognized field in config: " << fields[0] << '\n';
  }
  resize(ngram_size,
         input_vocab_size,
         output_vocab_size,
         input_embedding_dimension,
         num_hidden,
         output_embedding_dimension);
  set_activation_function(activation_function);
}

void model::readConfig(const string &filename)
{
  ifstream config_file(filename.c_str());
  if (!config_file)
  {
    cerr << "error: could not open config file " << filename << '\n';
    exit(1);
  }
  readConfig(config_file);
}

void model::read(const string &filename)
{
  vector<string> input_words;
  vector<string> output_words;
  read(filename, input_words, output_words);
}

void model::read(std::istream &file, std::ostream *log)
{
  vector<string> input_words;
  vector<string> output_words;
  read(file, input_words, output_words, log);
}


void model::read(const string &filename, std::vector<std::string> &input_words)
{
  ifstream file(filename.c_str());
  if (!file) throw runtime_error("Could not open file " + filename);
  read(file, input_words, NULL);
}

void model::read(std::istream &file, std::vector<std::string> &input_words, std::ostream *log)
{
  read(file, input_words, NULL, log);
}


void model::read(const string &filename, vector<string> &input_words, vector<string> &output_words)
{
  ifstream file(filename.c_str());
  if (!file) throw runtime_error("Could not open file " + filename);
  read(file, input_words, output_words);
}

void model::read(std::istream &file, vector<string> &input_words, vector<string> &output_words, std::ostream *log)
{
  read(file, input_words, &output_words, log);
}

void model::read(std::istream &file, vector<string> &input_words, vector<string> *output_words, std::ostream *log)
{
    param myParam;
    string line;

    while (getline(file, line))
    {
      rightTrim(line);
      string::size_type len = line.size();
      if (!len) continue;
      if (line[0] == '\\') {
        if (log) *log << "reading section " << line << "\n";
        if (line == "\\end")
          break;
        if (line == "\\config") {
          readConfig(file);
        } else if (line == "\\vocab") {
          input_words.clear();
          readWordsFile(file, input_words);
          if (log) *log << "vocab: " << input_words.size() << " words\n";
          if (output_words) *output_words = input_words;
        } else if (line == "\\input_vocab") {
          input_words.clear();
          readWordsFile(file, input_words);
          if (log) *log << "input_vocab: " << input_words.size() << " words\n";
        } else if (line == "\\output_vocab") {
          if (output_words) {
            output_words->clear();
            readWordsFile(file, *output_words);
            if (log) *log << "output_vocab: " << output_words->size() << " words\n";
          } else {
            if (log) *log << "skipping unexpected output_vocab section (not expected for neuralLM)\n";
            goto skip_section;
          }
        } else if (line == "\\input_embeddings")
          input_layer.read(file);
        else if (line == "\\hidden_weights 1")
          first_hidden_linear.read_weights(file);
        else if (line == "\\hidden_biases 1")
          first_hidden_linear.read_biases(file);
        else if (line == "\\hidden_weights 2")
          second_hidden_linear.read_weights(file);
        else if (line == "\\hidden_biases 2")
          second_hidden_linear.read_biases(file);
        else if (line == "\\output_weights")
          output_layer.read_weights(file);
        else if (line == "\\output_biases")
          output_layer.read_biases(file);
      } else {
	    cerr << "warning: unrecognized section: " << line << '\n';
	    if (log) *log << "warning: unrecognized section: " << line << '\n';
	    // skip over section
     skip_section:
	    while (getline(file, line)) {
          rightTrim(line);
          if (line.empty()) break;
        }
      }
    }
}


void model::write(const string &filename, const vector<string> &input_words, const vector<string> &output_words)
{
  write(filename, &input_words, &output_words);
}

void model::write(const string &filename, const vector<string> &words)
{
  write(filename, &words, NULL);
}

void model::write(const string &filename)
{
  write(filename, NULL, NULL);
}

void model::write(const string &filename, const vector<string> *input_pwords, const vector<string> *output_pwords)
{
  ofstream file(filename.c_str());
  if (!file) throw runtime_error("Could not open file " + filename);

  file << "\\config\n";
  file << "version 1\n";
  file << "ngram_size " << ngram_size << '\n';
  file << "input_vocab_size " << input_vocab_size << '\n';
  file << "output_vocab_size " << output_vocab_size << '\n';
  file << "input_embedding_dimension " << input_embedding_dimension << '\n';
  file << "num_hidden " << num_hidden << '\n';
  file << "output_embedding_dimension " << output_embedding_dimension << '\n';
  file << "activation_function " << activation_function_to_string(activation_function) << '\n';
  file << '\n';

  if (input_pwords)
  {
    file << "\\input_vocab\n";
    writeWordsFile(*input_pwords, file);
    file << '\n';
  }

  if (output_pwords)
  {
    file << "\\output_vocab\n";
    writeWordsFile(*output_pwords, file);
    file << '\n';
  }

  file << "\\input_embeddings\n";
  input_layer.write(file);
  file << '\n';

  file << "\\hidden_weights 1\n";
  first_hidden_linear.write_weights(file);
  file << '\n';

  file << "\\hidden_biases 1\n";
  first_hidden_linear.write_biases(file);
  file <<'\n';

  file << "\\hidden_weights 2\n";
  second_hidden_linear.write_weights(file);
  file << '\n';

  file << "\\hidden_biases 2\n";
  second_hidden_linear.write_biases(file);
  file << '\n';

  file << "\\output_weights\n";
  output_layer.write_weights(file);
  file << '\n';

  file << "\\output_biases\n";
  output_layer.write_biases(file);
  file << '\n';

  file << "\\end\n";
}


} // namespace nplm

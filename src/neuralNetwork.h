#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>

#include "util.h"
#include "model.h"
#include "propagator.h"
#include "neuralClasses.h"

typedef Eigen::Matrix<int,Eigen::Dynamic,1> EigenNgram;

namespace nplm
{


class neuralNetwork
{
 protected:
  boost::shared_ptr<model> m;
  int ngram_size_;
 private:
  bool normalization;
  double weight;

  propagator prop;

  std::size_t cache_size;
  Eigen::Matrix<int,Dynamic,Dynamic> cache_keys;
  std::vector<double> cache_values;
  int cache_lookups, cache_hits;

 public:

  neuralNetwork()
      : m(new model()),
        ngram_size_(),
        normalization(false),
        weight(1.),
        prop(*m, 1),
        cache_size(0)
  {
  }

  void set_normalization(bool value) { normalization = value; }
  void set_log_base(double value) { weight = 1./std::log(value); }

  // This must be called if the underlying model is resized.
  void resize() {
    ngram_size_ = m->ngram_size;
    if (cache_size)
    {
      cache_keys.resize(ngram_size_, cache_size);
      cache_keys.fill(-1);
    }
    prop.resize();
  }

  void set_width(int width)
  {
    prop.resize(width);
  }

  template <typename Derived>
  double lookup_ngram(const Eigen::MatrixBase<Derived> &ngram)
  {
    assert(ngram_size_ == m->ngram_size);
    assert(ngram.rows() == ngram_size_);
    assert(ngram.cols() == 1);

    std::size_t hash;
    if (cache_size)
    {
      // First look in cache
      hash = Eigen::hash_value(ngram) % cache_size; // defined in util.h
      cache_lookups++;
      if (cache_keys.col(hash) == ngram)
      {
        cache_hits++;
        return cache_values[hash];
      }
    }

    // Make sure that we're single threaded. Multithreading doesn't help,
    // and in some cases can hurt quite a lot
    int save_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    int save_eigen_threads = Eigen::nbThreads();
    Eigen::setNbThreads(1);
#ifdef __INTEL_MKL__
    int save_mkl_threads = mkl_get_max_threads();
    mkl_set_num_threads(1);
#endif

    prop.fProp(ngram.col(0));

    int output = ngram(ngram_size_-1, 0);
    double log_prob;

    start_timer(3);
    if (normalization)
    {
      Eigen::Matrix<double,Eigen::Dynamic,1> scores(m->output_vocab_size);
      if (prop.skip_hidden)
        prop.output_layer_node.param->fProp(prop.first_hidden_activation_node.fProp_matrix, scores);
      else
        prop.output_layer_node.param->fProp(prop.second_hidden_activation_node.fProp_matrix, scores);
      double logz = logsum(scores.col(0));
      log_prob = weight * (scores(output, 0) - logz);
    }
    else
    {
      if (prop.skip_hidden)
        log_prob = weight * prop.output_layer_node.param->fProp(prop.first_hidden_activation_node.fProp_matrix, output, 0);
      else
        log_prob = weight * prop.output_layer_node.param->fProp(prop.second_hidden_activation_node.fProp_matrix, output, 0);
    }
    stop_timer(3);

    if (cache_size)
    {
      // Update cache
      cache_keys.col(hash) = ngram;
      cache_values[hash] = log_prob;
    }

#ifdef __INTEL_MKL__
    mkl_set_num_threads(save_mkl_threads);
#endif
    Eigen::setNbThreads(save_eigen_threads);
    omp_set_num_threads(save_threads);

    return log_prob;
  }

  double lookup_ngram_start_null(const int *ngram_a, int n, int start, int null)
  {
    assert(n);
    assert(ngram_size_ == m->ngram_size);
    int want = ngram_size_;
    EigenNgram ngram(want);
    int missing = want - n;
    int i = 0;
    if (missing > 0) {
      int fill = ngram_a[0] == start ? start : null;
      for (; i < missing; ++i)
        ngram(i) = fill;
    } else
      ngram_a -= missing;
    for (; i < want; ++i)
      ngram(i) = *ngram_a++;
    return neuralNetwork::lookup_ngram(ngram);
  }

  // Look up many n-grams in parallel.
  template <typename DerivedA, typename DerivedB>
  void lookup_ngram(const Eigen::MatrixBase<DerivedA> &ngram, const Eigen::MatrixBase<DerivedB> &log_probs_const)
  {
    UNCONST(DerivedB, log_probs_const, log_probs);
    assert(ngram_size_ == m->ngram_size);
    assert(ngram.rows() == ngram_size_);
    //assert(ngram.cols() <= prop.get_minibatch_size());

    prop.fProp(ngram);

    if (normalization)
    {
      Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> scores(m->output_vocab_size, ngram.cols());
      if (prop.skip_hidden)
        prop.output_layer_node.param->fProp(prop.first_hidden_activation_node.fProp_matrix, scores);
      else
        prop.output_layer_node.param->fProp(prop.second_hidden_activation_node.fProp_matrix, scores);

      // And softmax and loss
      Matrix<double,Dynamic,Dynamic> output_probs(m->output_vocab_size, ngram.cols());
      double minibatch_log_likelihood;
      SoftmaxLogLoss().fProp(scores.leftCols(ngram.cols()), ngram.row(ngram_size_-1), output_probs, minibatch_log_likelihood);
      for (int j=0; j<ngram.cols(); j++)
      {
        int output = ngram(ngram_size_-1, j);
        log_probs(0, j) = weight * output_probs(output, j);
      }
    }
    else
    {
      for (int j=0; j<ngram.cols(); j++)
      {
        int output = ngram(ngram_size_-1, j);
        if (prop.skip_hidden)
          log_probs(0, j) = weight * prop.output_layer_node.param->fProp(prop.first_hidden_activation_node.fProp_matrix, output, j);
        else
          log_probs(0, j) = weight * prop.output_layer_node.param->fProp(prop.second_hidden_activation_node.fProp_matrix, output, j);
      }
    }
  }

  int get_order() const {
    assert(ngram_size_ == m->ngram_size);
    return ngram_size_;
  }

  void read(std::string const& filename, std::ostream *log = 0) {
    std::ifstream file(filename.c_str());
    if (!file) {
      std::cerr << "error: could not open neuralLM file " << filename << '\n';
      std::exit(1);
    }
    read(file, log);
  }

  void read(std::istream &file, std::ostream *log = 0)
  {
    m->read(file, log);
    resize();
    // this is faster but takes more memory
    //m->premultiply();
  }

  void set_cache(std::size_t cache_size)
  {
    assert(ngram_size_ == m->ngram_size);
    this->cache_size = cache_size;
    cache_keys.resize(ngram_size_, cache_size);
    cache_keys.fill(-1); // clears cache
    cache_values.resize(cache_size);
    cache_lookups = cache_hits = 0;
  }

  double cache_hit_rate()
  {
    return static_cast<double>(cache_hits)/cache_lookups;
  }

  void premultiply()
  {
    if (!m->premultiplied)
    {
      m->premultiply();
    }
  }

};

} // namespace nplm

#endif

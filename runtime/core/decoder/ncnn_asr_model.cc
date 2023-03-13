// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang, Di Wu)
//               2022 ZeXuan Li (lizexuan@huya.com)
//                    Xingchen Song(sxc19@mails.tsinghua.edu.cn)
//                    hamddct@gmail.com (Mddct)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "decoder/ncnn_asr_model.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "utils/string.h"

namespace wenet {

void NcnnAsrModel::InitEngineThreads(int num_threads) {

}

void NcnnAsrModel::Read(const std::string& model_dir) {
  try {
    InitNet(encoder_, model_dir + "/encoder.param", model_dir + "/encoder.bin");
    InitNet(ctc_, model_dir + "/ctc.param", model_dir + "/ctc.bin");
    //InitNet(rescore_, model_dir + "/rescore.param", model_dir + "/rescore.bin");
  } catch (std::exception const& e) {
    LOG(ERROR) << "error when load ncnn model: " << e.what();
    exit(0);
  }

  // Read model info
  const auto &blobs_encoder = encoder_->blobs();

  for (int32_t i = 0; i != blobs_encoder.size(); ++i) {
    const auto &b = blobs_encoder[i];
    LOG(INFO) << i << " : " << b.name;
    if(!strcmp(b.name.c_str(), "chunk")) {
      encoder_chunk_idx = i;
      continue;
    }
    if(!strcmp(b.name.c_str(), "pe_chunk")) {
      encoder_pe_chunk_idx = i;
      continue;
    }
    if(!strcmp(b.name.c_str(), "pe_att_cache")) {
      encoder_pe_att_cache_idx = i;
      continue;
    }
    if(!strcmp(b.name.c_str(), "att_cache")) {
      encoder_att_cache_idx = i;
      continue;
    }
    if(!strcmp(b.name.c_str(), "cnn_cache")) {
      encoder_cnn_cache_idx = i;
      continue;
    }
    if(!strcmp(b.name.c_str(), "output")) {
      encoder_output_idx = i;
      continue;
    }
    if(!strcmp(b.name.c_str(), "r_att_cache")) {
      encoder_r_att_cache_idx = i;
      continue;
    }
    if(!strcmp(b.name.c_str(), "r_cnn_cache")) {
      encoder_r_cnn_cache_idx = i;
      continue;
    }
  }

  const auto &blobs_ctc = ctc_->blobs();

  for (int32_t i = 0; i != blobs_ctc.size(); ++i) {
    const auto &b = blobs_ctc[i];
    if(!strcmp(b.name.c_str(), "ctchidden")) {
      ctc_hidden_idx = i;
      continue;
    }
    if(!strcmp(b.name.c_str(), "ctcprobs")) {
      ctc_probs_idx = i;
      continue;
    }
  }

  // Assign values for test
  encoder_output_size_ = 256;
  num_blocks_ = 12;
  cnn_module_kernel_ = 15;
  head_ = 4;
  subsampling_rate_ = 4;
  right_context_ = 6;
  sos_ = 11007;
  eos_ = 11007;
  is_bidirectional_decoder_ = false;
  chunk_size_ = 16;
  num_left_chunks_ = -1;
}

NcnnAsrModel::NcnnAsrModel(const NcnnAsrModel& other) {
  // metadatas
  encoder_output_size_ = other.encoder_output_size_;
  num_blocks_ = other.num_blocks_;
  head_ = other.head_;
  cnn_module_kernel_ = other.cnn_module_kernel_;
  right_context_ = other.right_context_;
  subsampling_rate_ = other.subsampling_rate_;
  sos_ = other.sos_;
  eos_ = other.eos_;
  is_bidirectional_decoder_ = other.is_bidirectional_decoder_;
  chunk_size_ = other.chunk_size_;
  num_left_chunks_ = other.num_left_chunks_;
  offset_ = other.offset_;

  // nets
  encoder_ = other.encoder_;
  ctc_= other.ctc_;
  rescore_ = other.rescore_;

  // blob indices
  encoder_chunk_idx = other.encoder_chunk_idx;
  encoder_pe_chunk_idx = other.encoder_pe_chunk_idx;
  encoder_pe_att_cache_idx = other.encoder_pe_att_cache_idx;
  encoder_att_cache_idx = other.encoder_att_cache_idx;
  encoder_cnn_cache_idx = other.encoder_cnn_cache_idx;

  encoder_output_idx = other.encoder_output_idx;
  encoder_r_att_cache_idx = other.encoder_r_att_cache_idx;
  encoder_r_cnn_cache_idx = other.encoder_r_cnn_cache_idx;

  ctc_hidden_idx = other.ctc_hidden_idx;
  ctc_probs_idx = other.ctc_probs_idx;
}

std::shared_ptr<AsrModel> NcnnAsrModel::Copy() const {
  auto asr_model = std::make_shared<NcnnAsrModel>(*this);
  // Reset the inner states for new decoding
  asr_model->Reset();
  return asr_model;
}

void NcnnAsrModel::Reset() {
  offset_ = 0;
  encoder_outs_.clear();
  cached_feature_.clear();

  // Reset att_cache
  if (num_left_chunks_ > 0) {
    int required_cache_size = chunk_size_ * num_left_chunks_;
    offset_ = required_cache_size;
    att_cache_.resize(num_blocks_ * head_ * required_cache_size *
                          encoder_output_size_ / head_ * 2,
                      0.0);
    const int64_t att_cache_shape[] = {num_blocks_, head_, required_cache_size,
                                       encoder_output_size_ / head_ * 2};
    att_cache_mat_ = ncnn::Mat(num_blocks_, required_cache_size, head_, encoder_output_size_ / head_ * 2);
  } else {
    att_cache_.resize(0, 0.0);
    const int64_t att_cache_shape[] = {num_blocks_, head_, 0,
                                       encoder_output_size_ / head_ * 2};
    att_cache_mat_ = ncnn::Mat(num_blocks_, 16, head_, encoder_output_size_ / head_ * 2);
  }

  // Reset cnn_cache
  cnn_cache_.resize(
      num_blocks_ * encoder_output_size_ * (cnn_module_kernel_ - 1), 0.0);
  const int64_t cnn_cache_shape[] = {num_blocks_, 1, encoder_output_size_,
                                     cnn_module_kernel_ - 1};
  cnn_cache_mat_ = ncnn::Mat(num_blocks_, encoder_output_size_, cnn_module_kernel_ - 1, 1);

  // Init positional encoding
  InitPositionalEncoding(5000, 256);
}

void NcnnAsrModel::InitNet(std::shared_ptr<ncnn::Net> &net, const std::string &param,
                    const std::string &bin) {
  net = std::make_shared<ncnn::Net>();
  if (net->load_param(param.c_str())) {
    LOG(ERROR) << "failed to load " << param;
    exit(-1);
  }

  if (net->load_model(bin.c_str())) {
    LOG(ERROR) << "failed to load " << bin;
    exit(-1);
  }
}

void NcnnAsrModel::ForwardEncoderFunc(
    const std::vector<std::vector<float>>& chunk_feats,
    std::vector<std::vector<float>>* out_prob) {
  // 1. Prepare ncnn required data, splice cached_feature_ and chunk_feats
  // chunk
  int num_frames = cached_feature_.size() + chunk_feats.size();
  const int feature_dim = chunk_feats[0].size();
  std::vector<float> feats;
  for (size_t i = 0; i < cached_feature_.size(); ++i) {
    feats.insert(feats.end(), cached_feature_[i].begin(),
                 cached_feature_[i].end());
  }
  for (size_t i = 0; i < chunk_feats.size(); ++i) {
    feats.insert(feats.end(), chunk_feats[i].begin(), chunk_feats[i].end());
  }
  const int64_t feats_shape[3] = {feature_dim, num_frames, 1};
  ncnn::Mat feat_mat(1, num_frames, feature_dim);
  memcpy(feat_mat.data, feats.data(), feats.size()*sizeof(float));
  // offset
  int64_t offset_int64 = static_cast<int64_t>(offset_);
  // required_cache_size
  int64_t required_cache_size = chunk_size_ * num_left_chunks_;
  // att_mask
  std::vector<uint8_t> att_mask(required_cache_size + chunk_size_, 1);
  printf("offset %lld, required_cache_size %lld, chunk_size_ %d\n", offset_int64, required_cache_size, chunk_size_);
  if (num_left_chunks_ > 0) {
    int chunk_idx = offset_ / chunk_size_ - num_left_chunks_;
    if (chunk_idx < num_left_chunks_) {
      for (int i = 0; i < (num_left_chunks_ - chunk_idx) * chunk_size_; ++i) {
        att_mask[i] = 0;
      }
    }
    const int64_t att_mask_shape[] = {1, 1, required_cache_size + chunk_size_};
  }
  // positional encoding
  const int64_t pe_chunk_shape[] = {1, chunk_size_, 256};
  std::vector<float> pe_chunk;
  for(size_t i=0; i<chunk_size_; i++) {
    for(size_t j=0; j<256; j++) {
      pe_chunk.emplace_back(pe_full[(offset_int64+i)*256+j]);
    }
  }
  ncnn::Mat pe_chunk_mat(256, chunk_size_, 1);
  memcpy(pe_chunk_mat.data, pe_chunk.data(), pe_chunk.size()*sizeof(float));

  const int64_t pe_att_cache_size = att_cache_mat_.h+chunk_size_;
  const int64_t pe_att_cache_shape[] = {1, pe_att_cache_size, 256};
  std::vector<float> pe_att_cache;
  for(size_t i=0; i<pe_att_cache_size; i++) {
    for(size_t j=0; j<256; j++) {
      pe_att_cache.emplace_back(pe_full[i*256+j]);
    }
  }
  ncnn::Mat pe_att_cache_mat(256, (int)pe_att_cache_size, 1);
  memcpy(pe_att_cache_mat.data, pe_att_cache.data(), pe_att_cache.size()*sizeof(float));

  // 2. Encoder chunk forward
  ncnn::Extractor encoder_ex = encoder_->create_extractor();
  encoder_ex.input(encoder_chunk_idx, feat_mat);
  encoder_ex.input(encoder_pe_chunk_idx, pe_chunk_mat);
  encoder_ex.input(encoder_pe_att_cache_idx, pe_att_cache_mat);
  encoder_ex.input(encoder_att_cache_idx, att_cache_mat_);
  encoder_ex.input(encoder_cnn_cache_idx, cnn_cache_mat_);

  ncnn::Mat encoder_output;
  ncnn::Mat encoder_r_att_cache;
  ncnn::Mat encoder_r_cnn_cache;

  encoder_ex.extract(encoder_output_idx, encoder_output);
  encoder_ex.extract(encoder_r_att_cache_idx, encoder_r_att_cache);
  encoder_ex.extract(encoder_r_cnn_cache_idx, encoder_r_cnn_cache);

  offset_ += encoder_output.h;
  att_cache_mat_ = encoder_r_att_cache;
  cnn_cache_mat_ = encoder_r_cnn_cache;

  //float* logp_data = ctc_ort_outputs[0].GetTensorMutableData<float>();
  //auto type_info = ctc_ort_outputs[0].GetTensorTypeAndShapeInfo();

  int num_outputs = 1;
  int output_dim = 11007;
  out_prob->resize(num_outputs);
  for (int i = 0; i < num_outputs; i++) {
    (*out_prob)[i].resize(output_dim);
    /*memcpy((*out_prob)[i].data(), 0.0 + i * output_dim,
           sizeof(float) * output_dim);*/
  }
}

float NcnnAsrModel::ComputeAttentionScore(const float* prob,
                                          const std::vector<int>& hyp, int eos,
                                          int decode_out_len) {
  float score = 0.0f;
  for (size_t j = 0; j < hyp.size(); ++j) {
    score += *(prob + j * decode_out_len + hyp[j]);
  }
  score += *(prob + hyp.size() * decode_out_len + eos);
  return score;
}

void NcnnAsrModel::AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                                      float reverse_weight,
                                      std::vector<float>* rescoring_score) {

}

void NcnnAsrModel::InitPositionalEncoding(int max_len, int d_model) {
  LOG(INFO) << "InitPositionalEncoding\n";

  std::vector<float> position;
  std::vector<float> div_term;

  for(int i=0; i<max_len; i++) {
    position.emplace_back((float)i);
  }

  for(int i=0; i<d_model; i+=2) {
    div_term.emplace_back(exp((float)i*(-log(10000.0)/d_model)));
  }

  for(int i=0; i<max_len; i++) {
    for(int j=0; j<div_term.size(); j++) {
      float pe_value = 0.0;
      pe_value = sin(position[i] * div_term[j]);
      pe_full.emplace_back(pe_value);
      pe_value = cos(position[i] * div_term[j]);
      pe_full.emplace_back(pe_value);
    }
  }

  LOG(INFO) << "InitPositionalEncoding done " << pe_full.size() << "\n";
}

}  // namespace wenet

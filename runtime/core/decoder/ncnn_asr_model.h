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

#ifndef DECODER_NCNN_ASR_MODEL_H_
#define DECODER_NCNN_ASR_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "net.h"  // NOLINT

#include "decoder/asr_model.h"
#include "utils/log.h"
#include "utils/utils.h"

namespace wenet {

class NcnnAsrModel : public AsrModel {
 public:
  static void InitEngineThreads(int num_threads = 1);

 public:
  NcnnAsrModel() = default;
  NcnnAsrModel(const NcnnAsrModel& other);
  void Read(const std::string& model_dir);
  void Reset() override;
  void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                          float reverse_weight,
                          std::vector<float>* rescoring_score) override;
  std::shared_ptr<AsrModel> Copy() const override;

 protected:
  void ForwardEncoderFunc(const std::vector<std::vector<float>>& chunk_feats,
                          std::vector<std::vector<float>>* ctc_prob) override;

  float ComputeAttentionScore(const float* prob, const std::vector<int>& hyp,
                              int eos, int decode_out_len);
  void InitNet(std::shared_ptr<ncnn::Net> &net, const std::string &param,
               const std::string &bin);

 private:
  int encoder_output_size_ = 0;
  int num_blocks_ = 0;
  int cnn_module_kernel_ = 0;
  int head_ = 0;
  std::shared_ptr<ncnn::Net> encoder_;
  std::shared_ptr<ncnn::Net> rescore_;
  std::shared_ptr<ncnn::Net> ctc_;

  // blob indices
  int32_t encoder_chunk_idx;
  int32_t encoder_pe_chunk_idx;
  int32_t encoder_pe_att_cache_idx;
  int32_t encoder_att_cache_idx;
  int32_t encoder_cnn_cache_idx;

  int32_t encoder_output_idx;
  int32_t encoder_r_att_cache_idx;
  int32_t encoder_r_cnn_cache_idx;

  int32_t ctc_hidden_idx;
  int32_t ctc_probs_idx;

  // caches
  ncnn::Mat att_cache_mat_;
  ncnn::Mat cnn_cache_mat_;
  std::vector<float> att_cache_;
  std::vector<float> cnn_cache_;
  std::vector<ncnn::Mat> encoder_outs_;

  //positional encoding
  std::vector<float> pe_full;

  void InitPositionalEncoding(int max_len, int d_model);
};

}  // namespace wenet

#endif  // DECODER_NCNN_ASR_MODEL_H_

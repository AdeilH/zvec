#include "zvec_c.h"

#include <cstdlib>
#include <cstring>
#include <functional>
#include <new>
#include <string>
#include <vector>

#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param.h>
#include <zvec/core/interface/index_param_builders.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/db/collection.h>
#include <zvec/db/config.h>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
#include <zvec/db/options.h>
#include <zvec/db/schema.h>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/ailego/utility/float_helper.h>

namespace {

thread_local std::string g_last_error;

zvec_status_t set_error(const std::string &msg,
                        zvec_status_t code = ZVEC_STATUS_ERR) {
  g_last_error = msg;
  return code;
}

char *dup_string(const std::string &s) {
  char *out = static_cast<char *>(std::malloc(s.size() + 1));
  if (!out) {
    return nullptr;
  }
  std::memcpy(out, s.c_str(), s.size() + 1);
  return out;
}

zvec::core_interface::MetricType to_metric(zvec_core_metric_type_t metric) {
  return static_cast<zvec::core_interface::MetricType>(metric);
}

zvec::core_interface::DataType to_data_type(zvec_core_data_type_t dt) {
  return static_cast<zvec::core_interface::DataType>(dt);
}

zvec::core_interface::StorageOptions::StorageType to_storage_type(
    zvec_core_storage_type_t t) {
  return static_cast<zvec::core_interface::StorageOptions::StorageType>(t);
}

zvec::DataType to_db_data_type(zvec_db_data_type_t t) {
  return static_cast<zvec::DataType>(t);
}

zvec::MetricType to_db_metric_type(zvec_db_metric_type_t t) {
  return static_cast<zvec::MetricType>(t);
}

zvec::QuantizeType to_db_quantize_type(zvec_db_quantize_type_t t) {
  return static_cast<zvec::QuantizeType>(t);
}

zvec::Operator to_db_operator(zvec_db_operator_t t) {
  return static_cast<zvec::Operator>(t);
}

zvec_status_t status_from_db_status(const zvec::Status &status) {
  if (status.ok()) {
    return ZVEC_STATUS_OK;
  }
  return set_error(status.message());
}

}  // namespace

struct zvec_core_param {
  zvec::core_interface::BaseIndexParam::Pointer ptr;
};

struct zvec_core_query_param {
  zvec::core_interface::BaseIndexQueryParam::Pointer ptr;
};

struct zvec_core_index {
  zvec::core_interface::Index::Pointer ptr;
};

struct zvec_core_search_result {
  std::vector<uint64_t> keys;
  std::vector<float> scores;
};

struct zvec_core_factory_object {
  std::shared_ptr<void> ptr;
};

struct zvec_db_schema {
  zvec::CollectionSchema::Ptr ptr;
};

struct zvec_db_field {
  zvec::FieldSchema::Ptr ptr;
};

struct zvec_db_index_params {
  zvec::IndexParams::Ptr ptr;
};

struct zvec_db_doc {
  zvec::Doc::Ptr ptr;
};

struct zvec_db_collection {
  zvec::Collection::Ptr ptr;
};

struct zvec_db_query_params {
  zvec::QueryParams::Ptr ptr;
};

struct zvec_db_query {
  zvec::VectorQuery query;
};

struct zvec_db_query_result {
  std::vector<zvec::Doc::Ptr> docs;
};

struct zvec_db_group_query {
  zvec::GroupByVectorQuery query;
};

struct zvec_db_group_result {
  std::vector<zvec::GroupResult> groups;
};

struct zvec_db_config {
  zvec::GlobalConfig::ConfigData data;
};

namespace {

template <typename T>
zvec_status_t get_field_value(const zvec_db_doc_t *doc, const char *field,
                              const char *type_name, T *out_value) {
  if (!doc || !doc->ptr || !field || !out_value) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  auto result = doc->ptr->get_field<T>(field);
  switch (result.status()) {
    case zvec::Doc::FieldGetStatus::SUCCESS:
      *out_value = result.value();
      return ZVEC_STATUS_OK;
    case zvec::Doc::FieldGetStatus::NOT_FOUND:
      return set_error(std::string("field not found: ") + field);
    case zvec::Doc::FieldGetStatus::IS_NULL:
      return set_error(std::string("field is null: ") + field);
    case zvec::Doc::FieldGetStatus::TYPE_MISMATCH:
      return set_error(std::string("field type mismatch for ") + field +
                       ", expected " + type_name);
  }
  return set_error("unknown field status");
}

template <typename T>
zvec_status_t copy_vector_out(const std::vector<T> &values, T **out_values,
                              size_t *out_count) {
  if (!out_values || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  if (values.empty()) {
    *out_values = nullptr;
    *out_count = 0;
    return ZVEC_STATUS_OK;
  }
  T *out = static_cast<T *>(std::malloc(values.size() * sizeof(T)));
  if (!out) {
    return set_error("failed to allocate output");
  }
  std::memcpy(out, values.data(), values.size() * sizeof(T));
  *out_values = out;
  *out_count = values.size();
  return ZVEC_STATUS_OK;
}

zvec_status_t copy_bool_vector_out(const std::vector<bool> &values,
                                   uint8_t **out_values,
                                   size_t *out_count) {
  if (!out_values || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  if (values.empty()) {
    *out_values = nullptr;
    *out_count = 0;
    return ZVEC_STATUS_OK;
  }
  uint8_t *out =
      static_cast<uint8_t *>(std::malloc(values.size() * sizeof(uint8_t)));
  if (!out) {
    return set_error("failed to allocate output");
  }
  for (size_t i = 0; i < values.size(); ++i) {
    out[i] = values[i] ? 1 : 0;
  }
  *out_values = out;
  *out_count = values.size();
  return ZVEC_STATUS_OK;
}

zvec_status_t copy_fp16_vector_out(
    const std::vector<zvec::ailego::Float16> &values,
    float **out_values, size_t *out_count) {
  if (!out_values || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  if (values.empty()) {
    *out_values = nullptr;
    *out_count = 0;
    return ZVEC_STATUS_OK;
  }
  float *out =
      static_cast<float *>(std::malloc(values.size() * sizeof(float)));
  if (!out) {
    return set_error("failed to allocate output");
  }
  for (size_t i = 0; i < values.size(); ++i) {
    out[i] = static_cast<float>(values[i]);
  }
  *out_values = out;
  *out_count = values.size();
  return ZVEC_STATUS_OK;
}

zvec_status_t copy_string_vector_out(const std::vector<std::string> &values,
                                     char ***out_items,
                                     size_t *out_count) {
  if (!out_items || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  if (values.empty()) {
    *out_items = nullptr;
    *out_count = 0;
    return ZVEC_STATUS_OK;
  }
  char **items = static_cast<char **>(
      std::calloc(values.size(), sizeof(char *)));
  if (!items) {
    return set_error("failed to allocate items");
  }
  for (size_t i = 0; i < values.size(); ++i) {
    items[i] = dup_string(values[i]);
    if (!items[i]) {
      for (size_t j = 0; j < i; ++j) {
        std::free(items[j]);
      }
      std::free(items);
      return set_error("failed to allocate item");
    }
  }
  *out_items = items;
  *out_count = values.size();
  return ZVEC_STATUS_OK;
}

zvec_status_t copy_binary_vector_out(const std::vector<std::string> &values,
                                     void ***out_items,
                                     size_t **out_sizes,
                                     size_t *out_count) {
  if (!out_items || !out_sizes || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  if (values.empty()) {
    *out_items = nullptr;
    *out_sizes = nullptr;
    *out_count = 0;
    return ZVEC_STATUS_OK;
  }
  void **items = static_cast<void **>(
      std::calloc(values.size(), sizeof(void *)));
  if (!items) {
    return set_error("failed to allocate items");
  }
  size_t *sizes = static_cast<size_t *>(
      std::calloc(values.size(), sizeof(size_t)));
  if (!sizes) {
    std::free(items);
    return set_error("failed to allocate sizes");
  }
  for (size_t i = 0; i < values.size(); ++i) {
    const auto &val = values[i];
    sizes[i] = val.size();
    if (sizes[i] == 0) {
      items[i] = nullptr;
      continue;
    }
    void *buf = std::malloc(sizes[i]);
    if (!buf) {
      for (size_t j = 0; j < i; ++j) {
        std::free(items[j]);
      }
      std::free(items);
      std::free(sizes);
      return set_error("failed to allocate item");
    }
    std::memcpy(buf, val.data(), sizes[i]);
    items[i] = buf;
  }
  *out_items = items;
  *out_sizes = sizes;
  *out_count = values.size();
  return ZVEC_STATUS_OK;
}

template <typename PtrT>
zvec_status_t make_factory_object(const PtrT &ptr,
                                  zvec_core_factory_object_t **out_obj,
                                  const char *what) {
  if (!out_obj) {
    return set_error("out_obj is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  if (!ptr) {
    return set_error(std::string(what) + " not found");
  }
  auto alias = std::shared_ptr<void>(ptr, ptr.get());
  *out_obj = new zvec_core_factory_object{std::move(alias)};
  return ZVEC_STATUS_OK;
}

}  // namespace

extern "C" const char *zvec_last_error(void) {
  return g_last_error.c_str();
}

extern "C" void zvec_clear_error(void) {
  g_last_error.clear();
}

extern "C" void zvec_free(void *p) {
  std::free(p);
}

extern "C" zvec_status_t zvec_core_param_create_hnsw(
    zvec_core_metric_type_t metric,
    zvec_core_data_type_t data_type,
    int dimension,
    int m,
    int ef_construction,
    int is_sparse,
    zvec_core_param_t **out_param) {
  if (!out_param) {
    return set_error("out_param is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto param = std::make_shared<zvec::core_interface::HNSWIndexParam>();
    param->metric_type = to_metric(metric);
    param->data_type = to_data_type(data_type);
    param->dimension = dimension;
    param->is_sparse = is_sparse != 0;
    param->m = m;
    param->ef_construction = ef_construction;
    auto *wrap = new zvec_core_param{param};
    *out_param = wrap;
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_core_param_create_ivf(
    zvec_core_metric_type_t metric,
    zvec_core_data_type_t data_type,
    int dimension,
    int nlist,
    int niters,
    int use_soar,
    int is_sparse,
    zvec_core_param_t **out_param) {
  if (!out_param) {
    return set_error("out_param is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto param = std::make_shared<zvec::core_interface::IVFIndexParam>();
    param->metric_type = to_metric(metric);
    param->data_type = to_data_type(data_type);
    param->dimension = dimension;
    param->is_sparse = is_sparse != 0;
    param->nlist = nlist;
    param->niters = niters;
    param->use_soar = use_soar != 0;
    auto *wrap = new zvec_core_param{param};
    *out_param = wrap;
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_core_param_create_flat(
    zvec_core_metric_type_t metric,
    zvec_core_data_type_t data_type,
    int dimension,
    int is_sparse,
    zvec_core_param_t **out_param) {
  if (!out_param) {
    return set_error("out_param is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto param = std::make_shared<zvec::core_interface::FlatIndexParam>();
    param->metric_type = to_metric(metric);
    param->data_type = to_data_type(data_type);
    param->dimension = dimension;
    param->is_sparse = is_sparse != 0;
    auto *wrap = new zvec_core_param{param};
    *out_param = wrap;
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_core_param_from_json(
    const char *json_str,
    zvec_core_param_t **out_param) {
  if (!json_str || !out_param) {
    return set_error("json_str or out_param is null",
                     ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto param =
        zvec::core_interface::IndexFactory::DeserializeIndexParamFromJson(
            json_str);
    if (!param) {
      return set_error("failed to deserialize index param from json");
    }
    auto *wrap = new zvec_core_param{param};
    *out_param = wrap;
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_core_param_to_json(
    const zvec_core_param_t *param,
    char **out_json) {
  if (!param || !param->ptr || !out_json) {
    return set_error("param or out_json is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    std::string json = param->ptr->SerializeToJson(false);
    char *dup = dup_string(json);
    if (!dup) {
      return set_error("failed to allocate json string");
    }
    *out_json = dup;
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" void zvec_core_param_destroy(zvec_core_param_t *param) {
  if (!param) {
    return;
  }
  delete param;
}

extern "C" zvec_status_t zvec_core_query_param_create_hnsw(
    int topk,
    int ef_search,
    int fetch_vector,
    float radius,
    int is_linear,
    zvec_core_query_param_t **out_param) {
  if (!out_param) {
    return set_error("out_param is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto qp = std::make_shared<zvec::core_interface::HNSWQueryParam>();
    qp->topk = topk;
    qp->ef_search = static_cast<uint32_t>(ef_search);
    qp->fetch_vector = fetch_vector != 0;
    qp->radius = radius;
    qp->is_linear = is_linear != 0;
    auto *wrap = new zvec_core_query_param{qp};
    *out_param = wrap;
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_core_query_param_create_ivf(
    int topk,
    int nprobe,
    int fetch_vector,
    float radius,
    int is_linear,
    zvec_core_query_param_t **out_param) {
  if (!out_param) {
    return set_error("out_param is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto qp = std::make_shared<zvec::core_interface::IVFQueryParam>();
    qp->topk = topk;
    qp->nprobe = nprobe;
    qp->fetch_vector = fetch_vector != 0;
    qp->radius = radius;
    qp->is_linear = is_linear != 0;
    auto *wrap = new zvec_core_query_param{qp};
    *out_param = wrap;
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_core_query_param_create_flat(
    int topk,
    int fetch_vector,
    float radius,
    int is_linear,
    zvec_core_query_param_t **out_param) {
  if (!out_param) {
    return set_error("out_param is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto qp = std::make_shared<zvec::core_interface::FlatQueryParam>();
    qp->topk = topk;
    qp->fetch_vector = fetch_vector != 0;
    qp->radius = radius;
    qp->is_linear = is_linear != 0;
    auto *wrap = new zvec_core_query_param{qp};
    *out_param = wrap;
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_core_query_param_to_json(
    const zvec_core_query_param_t *param,
    char **out_json) {
  if (!param || !param->ptr || !out_json) {
    return set_error("param or out_json is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    std::string json =
        zvec::core_interface::IndexFactory::QueryParamSerializeToJson<
            zvec::core_interface::BaseIndexQueryParam>(*param->ptr, false);
    char *dup = dup_string(json);
    if (!dup) {
      return set_error("failed to allocate json string");
    }
    *out_json = dup;
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_core_query_param_from_json_hnsw(
    const char *json_str,
    zvec_core_query_param_t **out_param) {
  if (!json_str || !out_param) {
    return set_error("json_str or out_param is null",
                     ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto param =
        zvec::core_interface::IndexFactory::QueryParamDeserializeFromJson<
            zvec::core_interface::HNSWQueryParam>(json_str);
    if (!param) {
      return set_error("failed to deserialize hnsw query param from json");
    }
    *out_param = new zvec_core_query_param{param};
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_core_query_param_from_json_ivf(
    const char *json_str,
    zvec_core_query_param_t **out_param) {
  if (!json_str || !out_param) {
    return set_error("json_str or out_param is null",
                     ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto param =
        zvec::core_interface::IndexFactory::QueryParamDeserializeFromJson<
            zvec::core_interface::IVFQueryParam>(json_str);
    if (!param) {
      return set_error("failed to deserialize ivf query param from json");
    }
    *out_param = new zvec_core_query_param{param};
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_core_query_param_from_json_flat(
    const char *json_str,
    zvec_core_query_param_t **out_param) {
  if (!json_str || !out_param) {
    return set_error("json_str or out_param is null",
                     ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto param =
        zvec::core_interface::IndexFactory::QueryParamDeserializeFromJson<
            zvec::core_interface::FlatQueryParam>(json_str);
    if (!param) {
      return set_error("failed to deserialize flat query param from json");
    }
    *out_param = new zvec_core_query_param{param};
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" void zvec_core_query_param_destroy(zvec_core_query_param_t *param) {
  if (!param) {
    return;
  }
  delete param;
}

extern "C" zvec_status_t zvec_core_index_create(
    const zvec_core_param_t *param,
    zvec_core_index_t **out_index) {
  if (!param || !param->ptr || !out_index) {
    return set_error("param or out_index is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto idx =
        zvec::core_interface::IndexFactory::CreateAndInitIndex(*param->ptr);
    if (!idx) {
      return set_error("failed to create index");
    }
    auto *wrap = new zvec_core_index{idx};
    *out_index = wrap;
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_core_index_open(
    zvec_core_index_t *index,
    const char *path,
    zvec_core_storage_type_t storage_type,
    int create_new,
    int read_only) {
  if (!index || !index->ptr || !path) {
    return set_error("index or path is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    zvec::core_interface::StorageOptions options;
    options.type = to_storage_type(storage_type);
    options.create_new = create_new != 0;
    options.read_only = read_only != 0;
    int ret = index->ptr->Open(path, options);
    if (ret != 0) {
      return set_error("index open failed");
    }
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_core_index_close(zvec_core_index_t *index) {
  if (!index || !index->ptr) {
    return set_error("index is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    int ret = index->ptr->Close();
    if (ret != 0) {
      return set_error("index close failed");
    }
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_core_index_flush(zvec_core_index_t *index) {
  if (!index || !index->ptr) {
    return set_error("index is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    int ret = index->ptr->Flush();
    if (ret != 0) {
      return set_error("index flush failed");
    }
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_core_index_train(zvec_core_index_t *index) {
  if (!index || !index->ptr) {
    return set_error("index is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    int ret = index->ptr->Train();
    if (ret != 0) {
      return set_error("index train failed");
    }
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" uint32_t zvec_core_index_doc_count(
    const zvec_core_index_t *index) {
  if (!index || !index->ptr) {
    set_error("index is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return index->ptr->GetDocCount();
}

extern "C" void zvec_core_index_destroy(zvec_core_index_t *index) {
  if (!index) {
    return;
  }
  delete index;
}

extern "C" zvec_status_t zvec_core_index_add_dense(
    zvec_core_index_t *index,
    const void *data,
    size_t element_count,
    uint32_t doc_id) {
  if (!index || !index->ptr || !data || element_count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    zvec::core_interface::VectorData v;
    v.vector = zvec::core_interface::DenseVector{data};
    int ret = index->ptr->Add(v, doc_id);
    if (ret != 0) {
      return set_error("index add failed");
    }
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_core_index_add_sparse(
    zvec_core_index_t *index,
    const uint32_t *indices,
    const void *values,
    uint32_t count,
    uint32_t doc_id) {
  if (!index || !index->ptr || !indices || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    zvec::core_interface::SparseVector sp;
    sp.count = count;
    sp.indices = indices;
    sp.values = values;
    zvec::core_interface::VectorData v;
    v.vector = sp;
    int ret = index->ptr->Add(v, doc_id);
    if (ret != 0) {
      return set_error("index add failed");
    }
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_core_index_search_dense(
    zvec_core_index_t *index,
    const void *query,
    size_t element_count,
    const zvec_core_query_param_t *param,
    zvec_core_search_result_t **out_result) {
  if (!index || !index->ptr || !query || element_count == 0 || !param ||
      !param->ptr || !out_result) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    zvec::core_interface::VectorData v;
    v.vector = zvec::core_interface::DenseVector{query};
    zvec::core_interface::SearchResult result;
    int ret = index->ptr->Search(v, param->ptr, &result);
    if (ret != 0) {
      return set_error("index search failed");
    }
    auto *wrap = new zvec_core_search_result;
    wrap->keys.reserve(result.doc_list_.size());
    wrap->scores.reserve(result.doc_list_.size());
    for (const auto &doc : result.doc_list_) {
      wrap->keys.push_back(doc.key());
      wrap->scores.push_back(doc.score());
    }
    *out_result = wrap;
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_core_index_search_sparse(
    zvec_core_index_t *index,
    const uint32_t *indices,
    const void *values,
    uint32_t count,
    const zvec_core_query_param_t *param,
    zvec_core_search_result_t **out_result) {
  if (!index || !index->ptr || !indices || !values || count == 0 || !param ||
      !param->ptr || !out_result) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    zvec::core_interface::SparseVector sp;
    sp.count = count;
    sp.indices = indices;
    sp.values = values;
    zvec::core_interface::VectorData v;
    v.vector = sp;
    zvec::core_interface::SearchResult result;
    int ret = index->ptr->Search(v, param->ptr, &result);
    if (ret != 0) {
      return set_error("index search failed");
    }
    auto *wrap = new zvec_core_search_result;
    wrap->keys.reserve(result.doc_list_.size());
    wrap->scores.reserve(result.doc_list_.size());
    for (const auto &doc : result.doc_list_) {
      wrap->keys.push_back(doc.key());
      wrap->scores.push_back(doc.score());
    }
    *out_result = wrap;
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_core_index_fetch_dense(
    zvec_core_index_t *index,
    uint32_t doc_id,
    void **out_data,
    size_t *out_bytes) {
  if (!index || !index->ptr || !out_data || !out_bytes) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    zvec::core_interface::VectorDataBuffer buffer;
    int ret = index->ptr->Fetch(doc_id, &buffer);
    if (ret != 0) {
      return set_error("index fetch failed");
    }
    auto dense =
        std::get_if<zvec::core_interface::DenseVectorBuffer>(
            &buffer.vector_buffer);
    if (!dense) {
      return set_error("index fetch returned sparse data");
    }
    if (dense->data.empty()) {
      *out_data = nullptr;
      *out_bytes = 0;
      return ZVEC_STATUS_OK;
    }
    void *out = std::malloc(dense->data.size());
    if (!out) {
      return set_error("failed to allocate output");
    }
    std::memcpy(out, dense->data.data(), dense->data.size());
    *out_data = out;
    *out_bytes = dense->data.size();
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_core_index_fetch_sparse(
    zvec_core_index_t *index,
    uint32_t doc_id,
    uint32_t **out_indices,
    void **out_values,
    uint32_t *out_count,
    size_t *out_value_bytes) {
  if (!index || !index->ptr || !out_indices || !out_values || !out_count ||
      !out_value_bytes) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    zvec::core_interface::VectorDataBuffer buffer;
    int ret = index->ptr->Fetch(doc_id, &buffer);
    if (ret != 0) {
      return set_error("index fetch failed");
    }
    auto sparse =
        std::get_if<zvec::core_interface::SparseVectorBuffer>(
            &buffer.vector_buffer);
    if (!sparse) {
      return set_error("index fetch returned dense data");
    }
    if (sparse->count == 0) {
      *out_indices = nullptr;
      *out_values = nullptr;
      *out_count = 0;
      *out_value_bytes = 0;
      return ZVEC_STATUS_OK;
    }
    if (sparse->indices.size() != sparse->count * sizeof(uint32_t)) {
      return set_error("sparse index buffer size mismatch");
    }
    uint32_t *indices = static_cast<uint32_t *>(
        std::malloc(sparse->indices.size()));
    if (!indices) {
      return set_error("failed to allocate indices");
    }
    void *values = std::malloc(sparse->values.size());
    if (!values) {
      std::free(indices);
      return set_error("failed to allocate values");
    }
    std::memcpy(indices, sparse->indices.data(), sparse->indices.size());
    std::memcpy(values, sparse->values.data(), sparse->values.size());
    *out_indices = indices;
    *out_values = values;
    *out_count = sparse->count;
    *out_value_bytes = sparse->values.size();
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" int zvec_core_index_is_trained(const zvec_core_index_t *index) {
  if (!index || !index->ptr) {
    set_error("index is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return index->ptr->IsTrained() ? 1 : 0;
}

extern "C" zvec_status_t zvec_core_index_get_param_json(
    const zvec_core_index_t *index,
    char **out_json) {
  if (!index || !index->ptr || !out_json) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto param = index->ptr->GetParam();
    if (!param) {
      return set_error("failed to get index param");
    }
    std::string json = param->SerializeToJson(false);
    char *dup = dup_string(json);
    if (!dup) {
      return set_error("failed to allocate json string");
    }
    *out_json = dup;
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" size_t zvec_core_search_result_size(
    const zvec_core_search_result_t *result) {
  if (!result) {
    set_error("result is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return result->keys.size();
}

extern "C" zvec_status_t zvec_core_search_result_get(
    const zvec_core_search_result_t *result,
    size_t idx,
    uint64_t *out_key,
    float *out_score) {
  if (!result || !out_key || !out_score) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  if (idx >= result->keys.size()) {
    return set_error("index out of range", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  *out_key = result->keys[idx];
  *out_score = result->scores[idx];
  return ZVEC_STATUS_OK;
}

extern "C" void zvec_core_search_result_destroy(
    zvec_core_search_result_t *result) {
  if (!result) {
    return;
  }
  delete result;
}

extern "C" zvec_status_t zvec_core_factory_create_metric(
    const char *name,
    zvec_core_factory_object_t **out_obj) {
  if (!name) {
    return set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return make_factory_object(zvec::core::IndexFactory::CreateMetric(name),
                             out_obj, "metric");
}

extern "C" int zvec_core_factory_has_metric(const char *name) {
  if (!name) {
    set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return zvec::core::IndexFactory::HasMetric(name) ? 1 : 0;
}

extern "C" zvec_status_t zvec_core_factory_all_metrics(
    char ***out_names,
    size_t *out_count) {
  if (!out_names || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return copy_string_vector_out(zvec::core::IndexFactory::AllMetrics(),
                                out_names, out_count);
}

extern "C" zvec_status_t zvec_core_factory_create_logger(
    const char *name,
    zvec_core_factory_object_t **out_obj) {
  if (!name) {
    return set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return make_factory_object(zvec::core::IndexFactory::CreateLogger(name),
                             out_obj, "logger");
}

extern "C" int zvec_core_factory_has_logger(const char *name) {
  if (!name) {
    set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return zvec::core::IndexFactory::HasLogger(name) ? 1 : 0;
}

extern "C" zvec_status_t zvec_core_factory_all_loggers(
    char ***out_names,
    size_t *out_count) {
  if (!out_names || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return copy_string_vector_out(zvec::core::IndexFactory::AllLoggers(),
                                out_names, out_count);
}

extern "C" zvec_status_t zvec_core_factory_create_dumper(
    const char *name,
    zvec_core_factory_object_t **out_obj) {
  if (!name) {
    return set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return make_factory_object(zvec::core::IndexFactory::CreateDumper(name),
                             out_obj, "dumper");
}

extern "C" int zvec_core_factory_has_dumper(const char *name) {
  if (!name) {
    set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return zvec::core::IndexFactory::HasDumper(name) ? 1 : 0;
}

extern "C" zvec_status_t zvec_core_factory_all_dumpers(
    char ***out_names,
    size_t *out_count) {
  if (!out_names || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return copy_string_vector_out(zvec::core::IndexFactory::AllDumpers(),
                                out_names, out_count);
}

extern "C" zvec_status_t zvec_core_factory_create_container(
    const char *name,
    zvec_core_factory_object_t **out_obj) {
  if (!name) {
    return set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return make_factory_object(zvec::core::IndexFactory::CreateContainer(name),
                             out_obj, "container");
}

extern "C" int zvec_core_factory_has_container(const char *name) {
  if (!name) {
    set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return zvec::core::IndexFactory::HasContainer(name) ? 1 : 0;
}

extern "C" zvec_status_t zvec_core_factory_all_containers(
    char ***out_names,
    size_t *out_count) {
  if (!out_names || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return copy_string_vector_out(zvec::core::IndexFactory::AllContainers(),
                                out_names, out_count);
}

extern "C" zvec_status_t zvec_core_factory_create_storage(
    const char *name,
    zvec_core_factory_object_t **out_obj) {
  if (!name) {
    return set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return make_factory_object(zvec::core::IndexFactory::CreateStorage(name),
                             out_obj, "storage");
}

extern "C" int zvec_core_factory_has_storage(const char *name) {
  if (!name) {
    set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return zvec::core::IndexFactory::HasStorage(name) ? 1 : 0;
}

extern "C" zvec_status_t zvec_core_factory_all_storages(
    char ***out_names,
    size_t *out_count) {
  if (!out_names || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return copy_string_vector_out(zvec::core::IndexFactory::AllStorages(),
                                out_names, out_count);
}

extern "C" zvec_status_t zvec_core_factory_create_converter(
    const char *name,
    zvec_core_factory_object_t **out_obj) {
  if (!name) {
    return set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return make_factory_object(zvec::core::IndexFactory::CreateConverter(name),
                             out_obj, "converter");
}

extern "C" int zvec_core_factory_has_converter(const char *name) {
  if (!name) {
    set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return zvec::core::IndexFactory::HasConverter(name) ? 1 : 0;
}

extern "C" zvec_status_t zvec_core_factory_all_converters(
    char ***out_names,
    size_t *out_count) {
  if (!out_names || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return copy_string_vector_out(zvec::core::IndexFactory::AllConverters(),
                                out_names, out_count);
}

extern "C" zvec_status_t zvec_core_factory_create_reformer(
    const char *name,
    zvec_core_factory_object_t **out_obj) {
  if (!name) {
    return set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return make_factory_object(zvec::core::IndexFactory::CreateReformer(name),
                             out_obj, "reformer");
}

extern "C" int zvec_core_factory_has_reformer(const char *name) {
  if (!name) {
    set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return zvec::core::IndexFactory::HasReformer(name) ? 1 : 0;
}

extern "C" zvec_status_t zvec_core_factory_all_reformers(
    char ***out_names,
    size_t *out_count) {
  if (!out_names || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return copy_string_vector_out(zvec::core::IndexFactory::AllReformers(),
                                out_names, out_count);
}

extern "C" zvec_status_t zvec_core_factory_create_trainer(
    const char *name,
    zvec_core_factory_object_t **out_obj) {
  if (!name) {
    return set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return make_factory_object(zvec::core::IndexFactory::CreateTrainer(name),
                             out_obj, "trainer");
}

extern "C" int zvec_core_factory_has_trainer(const char *name) {
  if (!name) {
    set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return zvec::core::IndexFactory::HasTrainer(name) ? 1 : 0;
}

extern "C" zvec_status_t zvec_core_factory_all_trainers(
    char ***out_names,
    size_t *out_count) {
  if (!out_names || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return copy_string_vector_out(zvec::core::IndexFactory::AllTrainers(),
                                out_names, out_count);
}

extern "C" zvec_status_t zvec_core_factory_create_builder(
    const char *name,
    zvec_core_factory_object_t **out_obj) {
  if (!name) {
    return set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return make_factory_object(zvec::core::IndexFactory::CreateBuilder(name),
                             out_obj, "builder");
}

extern "C" int zvec_core_factory_has_builder(const char *name) {
  if (!name) {
    set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return zvec::core::IndexFactory::HasBuilder(name) ? 1 : 0;
}

extern "C" zvec_status_t zvec_core_factory_all_builders(
    char ***out_names,
    size_t *out_count) {
  if (!out_names || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return copy_string_vector_out(zvec::core::IndexFactory::AllBuilders(),
                                out_names, out_count);
}

extern "C" zvec_status_t zvec_core_factory_create_searcher(
    const char *name,
    zvec_core_factory_object_t **out_obj) {
  if (!name) {
    return set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return make_factory_object(zvec::core::IndexFactory::CreateSearcher(name),
                             out_obj, "searcher");
}

extern "C" int zvec_core_factory_has_searcher(const char *name) {
  if (!name) {
    set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return zvec::core::IndexFactory::HasSearcher(name) ? 1 : 0;
}

extern "C" zvec_status_t zvec_core_factory_all_searchers(
    char ***out_names,
    size_t *out_count) {
  if (!out_names || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return copy_string_vector_out(zvec::core::IndexFactory::AllSearchers(),
                                out_names, out_count);
}

extern "C" zvec_status_t zvec_core_factory_create_streamer(
    const char *name,
    zvec_core_factory_object_t **out_obj) {
  if (!name) {
    return set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return make_factory_object(zvec::core::IndexFactory::CreateStreamer(name),
                             out_obj, "streamer");
}

extern "C" int zvec_core_factory_has_streamer(const char *name) {
  if (!name) {
    set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return zvec::core::IndexFactory::HasStreamer(name) ? 1 : 0;
}

extern "C" zvec_status_t zvec_core_factory_all_streamers(
    char ***out_names,
    size_t *out_count) {
  if (!out_names || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return copy_string_vector_out(zvec::core::IndexFactory::AllStreamers(),
                                out_names, out_count);
}

extern "C" zvec_status_t zvec_core_factory_create_reducer(
    const char *name,
    zvec_core_factory_object_t **out_obj) {
  if (!name) {
    return set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return make_factory_object(zvec::core::IndexFactory::CreateReducer(name),
                             out_obj, "reducer");
}

extern "C" int zvec_core_factory_has_reducer(const char *name) {
  if (!name) {
    set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return zvec::core::IndexFactory::HasReducer(name) ? 1 : 0;
}

extern "C" zvec_status_t zvec_core_factory_all_reducers(
    char ***out_names,
    size_t *out_count) {
  if (!out_names || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return copy_string_vector_out(zvec::core::IndexFactory::AllReducers(),
                                out_names, out_count);
}

extern "C" zvec_status_t zvec_core_factory_create_cluster(
    const char *name,
    zvec_core_factory_object_t **out_obj) {
  if (!name) {
    return set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return make_factory_object(zvec::core::IndexFactory::CreateCluster(name),
                             out_obj, "cluster");
}

extern "C" int zvec_core_factory_has_cluster(const char *name) {
  if (!name) {
    set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return zvec::core::IndexFactory::HasCluster(name) ? 1 : 0;
}

extern "C" zvec_status_t zvec_core_factory_all_clusters(
    char ***out_names,
    size_t *out_count) {
  if (!out_names || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return copy_string_vector_out(zvec::core::IndexFactory::AllClusters(),
                                out_names, out_count);
}

extern "C" zvec_status_t zvec_core_factory_create_streamer_reducer(
    const char *name,
    zvec_core_factory_object_t **out_obj) {
  if (!name) {
    return set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return make_factory_object(
      zvec::core::IndexFactory::CreateStreamerReducer(name),
      out_obj, "streamer_reducer");
}

extern "C" int zvec_core_factory_has_streamer_reducer(const char *name) {
  if (!name) {
    set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return zvec::core::IndexFactory::HasStreamerReducer(name) ? 1 : 0;
}

extern "C" zvec_status_t zvec_core_factory_all_streamer_reducers(
    char ***out_names,
    size_t *out_count) {
  if (!out_names || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return copy_string_vector_out(
      zvec::core::IndexFactory::AllStreamerReducers(), out_names, out_count);
}

extern "C" zvec_status_t zvec_core_factory_create_refiner(
    const char *name,
    zvec_core_factory_object_t **out_obj) {
  if (!name) {
    return set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return make_factory_object(zvec::core::IndexFactory::CreateRefiner(name),
                             out_obj, "refiner");
}

extern "C" int zvec_core_factory_has_refiner(const char *name) {
  if (!name) {
    set_error("name is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return zvec::core::IndexFactory::HasRefiner(name) ? 1 : 0;
}

extern "C" zvec_status_t zvec_core_factory_all_refiners(
    char ***out_names,
    size_t *out_count) {
  if (!out_names || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return copy_string_vector_out(zvec::core::IndexFactory::AllRefiners(),
                                out_names, out_count);
}

extern "C" void zvec_core_factory_object_destroy(
    zvec_core_factory_object_t *obj) {
  if (!obj) {
    return;
  }
  delete obj;
}

extern "C" void zvec_core_factory_string_array_free(
    char **items, size_t count) {
  if (!items) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    std::free(items[i]);
  }
  std::free(items);
}

extern "C" zvec_status_t zvec_db_schema_create(
    const char *name,
    zvec_db_schema_t **out) {
  if (!name || !out) {
    return set_error("name or out is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto schema = std::make_shared<zvec::CollectionSchema>(name);
    *out = new zvec_db_schema{schema};
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" const char *zvec_db_status_default_message(
    zvec_db_status_code_t code) {
  return zvec::GetDefaultMessage(
      static_cast<zvec::StatusCode>(code));
}

extern "C" zvec_status_t zvec_db_schema_set_max_docs_per_segment(
    zvec_db_schema_t *schema,
    uint64_t max_docs) {
  if (!schema || !schema->ptr) {
    return set_error("schema is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  schema->ptr->set_max_doc_count_per_segment(max_docs);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_schema_add_field(
    zvec_db_schema_t *schema,
    const zvec_db_field_t *field) {
  if (!schema || !schema->ptr || !field || !field->ptr) {
    return set_error("schema or field is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return status_from_db_status(schema->ptr->add_field(field->ptr));
}

extern "C" int zvec_db_schema_has_field(
    const zvec_db_schema_t *schema,
    const char *name) {
  if (!schema || !schema->ptr || !name) {
    set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return schema->ptr->has_field(name) ? 1 : 0;
}

extern "C" zvec_status_t zvec_db_schema_drop_field(
    zvec_db_schema_t *schema,
    const char *name) {
  if (!schema || !schema->ptr || !name) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return status_from_db_status(schema->ptr->drop_field(name));
}

extern "C" zvec_status_t zvec_db_schema_alter_field(
    zvec_db_schema_t *schema,
    const char *name,
    const zvec_db_field_t *new_field) {
  if (!schema || !schema->ptr || !name || !new_field || !new_field->ptr) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return status_from_db_status(
      schema->ptr->alter_field(name, new_field->ptr));
}

extern "C" zvec_status_t zvec_db_schema_add_index(
    zvec_db_schema_t *schema,
    const char *field,
    const zvec_db_index_params_t *params) {
  if (!schema || !schema->ptr || !field || !params || !params->ptr) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return status_from_db_status(schema->ptr->add_index(field, params->ptr));
}

extern "C" zvec_status_t zvec_db_schema_drop_index(
    zvec_db_schema_t *schema,
    const char *field) {
  if (!schema || !schema->ptr || !field) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return status_from_db_status(schema->ptr->drop_index(field));
}

extern "C" int zvec_db_schema_has_index(
    const zvec_db_schema_t *schema,
    const char *field) {
  if (!schema || !schema->ptr || !field) {
    set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return schema->ptr->has_index(field) ? 1 : 0;
}

extern "C" zvec_status_t zvec_db_schema_get_name(
    const zvec_db_schema_t *schema,
    char **out_name) {
  if (!schema || !schema->ptr || !out_name) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  char *dup = dup_string(schema->ptr->name());
  if (!dup) {
    return set_error("failed to allocate string");
  }
  *out_name = dup;
  return ZVEC_STATUS_OK;
}

extern "C" uint64_t zvec_db_schema_get_max_docs_per_segment(
    const zvec_db_schema_t *schema) {
  if (!schema || !schema->ptr) {
    set_error("schema is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return schema->ptr->max_doc_count_per_segment();
}

extern "C" zvec_status_t zvec_db_schema_to_string(
    const zvec_db_schema_t *schema,
    char **out_string) {
  if (!schema || !schema->ptr || !out_string) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  char *dup = dup_string(schema->ptr->to_string());
  if (!dup) {
    return set_error("failed to allocate string");
  }
  *out_string = dup;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_schema_to_formatted_string(
    const zvec_db_schema_t *schema,
    int indent_level,
    char **out_string) {
  if (!schema || !schema->ptr || !out_string) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  char *dup = dup_string(schema->ptr->to_string_formatted(indent_level));
  if (!dup) {
    return set_error("failed to allocate string");
  }
  *out_string = dup;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_schema_validate(
    const zvec_db_schema_t *schema) {
  if (!schema || !schema->ptr) {
    return set_error("schema is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return status_from_db_status(schema->ptr->validate());
}

extern "C" size_t zvec_db_schema_field_count(
    const zvec_db_schema_t *schema) {
  if (!schema || !schema->ptr) {
    set_error("schema is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return schema->ptr->fields().size();
}

extern "C" zvec_status_t zvec_db_schema_get_field(
    const zvec_db_schema_t *schema,
    size_t index,
    zvec_db_field_t **out_field) {
  if (!schema || !schema->ptr || !out_field) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  *out_field = nullptr;
  auto fields = schema->ptr->fields();
  if (index >= fields.size()) {
    return set_error("index out of range", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  auto field = std::make_shared<zvec::FieldSchema>(*fields[index]);
  *out_field = new zvec_db_field{field};
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_schema_get_field_by_name(
    const zvec_db_schema_t *schema,
    const char *name,
    zvec_db_field_t **out_field) {
  if (!schema || !schema->ptr || !name || !out_field) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  *out_field = nullptr;
  const auto *field = schema->ptr->get_field(name);
  if (!field) {
    return set_error("field not found");
  }
  auto copy = std::make_shared<zvec::FieldSchema>(*field);
  *out_field = new zvec_db_field{copy};
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_schema_field_names(
    const zvec_db_schema_t *schema,
    char ***out_names,
    size_t *out_count) {
  if (!schema || !schema->ptr || !out_names || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  auto names = schema->ptr->all_field_names();
  return copy_string_vector_out(names, out_names, out_count);
}

extern "C" size_t zvec_db_schema_sizeof(void) {
    return sizeof(zvec_db_schema_t);
}

extern "C" void zvec_db_schema_destroy(zvec_db_schema_t *schema) {
  if (!schema) {
    return;
  }
  delete schema;
}

extern "C" zvec_status_t zvec_db_field_create_scalar(
    const char *name,
    zvec_db_data_type_t data_type,
    int nullable,
    const zvec_db_index_params_t *index_params,
    zvec_db_field_t **out_field) {
  if (!name || !out_field) {
    return set_error("name or out_field is null",
                     ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    zvec::IndexParams::Ptr idx =
        index_params ? index_params->ptr : nullptr;
    auto field = std::make_shared<zvec::FieldSchema>(
        name, to_db_data_type(data_type), nullable != 0, idx);
    *out_field = new zvec_db_field{field};
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_db_field_create_vector(
    const char *name,
    zvec_db_data_type_t data_type,
    uint32_t dimension,
    int nullable,
    const zvec_db_index_params_t *index_params,
    zvec_db_field_t **out_field) {
  if (!name || !out_field) {
    return set_error("name or out_field is null",
                     ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    zvec::IndexParams::Ptr idx =
        index_params ? index_params->ptr : nullptr;
    auto field = std::make_shared<zvec::FieldSchema>(
        name, to_db_data_type(data_type), dimension, nullable != 0, idx);
    *out_field = new zvec_db_field{field};
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_db_field_create_sparse_vector(
    const char *name,
    zvec_db_data_type_t data_type,
    uint32_t dimension,
    int nullable,
    const zvec_db_index_params_t *index_params,
    zvec_db_field_t **out_field) {
  if (data_type != ZVEC_DB_DT_SPARSE_VECTOR_FP16 &&
      data_type != ZVEC_DB_DT_SPARSE_VECTOR_FP32) {
    return set_error("data_type is not sparse vector",
                     ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return zvec_db_field_create_vector(
      name, data_type, dimension, nullable, index_params, out_field);
}

extern "C" zvec_status_t zvec_db_field_get_name(
    const zvec_db_field_t *field,
    char **out_name) {
  if (!field || !field->ptr || !out_name) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  char *dup = dup_string(field->ptr->name());
  if (!dup) {
    return set_error("failed to allocate string");
  }
  *out_name = dup;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_field_get_data_type(
    const zvec_db_field_t *field,
    zvec_db_data_type_t *out_type) {
  if (!field || !field->ptr || !out_type) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  *out_type = static_cast<zvec_db_data_type_t>(field->ptr->data_type());
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_field_get_dimension(
    const zvec_db_field_t *field,
    uint32_t *out_dimension) {
  if (!field || !field->ptr || !out_dimension) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  *out_dimension = field->ptr->dimension();
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_field_get_nullable(
    const zvec_db_field_t *field,
    int *out_nullable) {
  if (!field || !field->ptr || !out_nullable) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  *out_nullable = field->ptr->nullable() ? 1 : 0;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_field_get_index_type(
    const zvec_db_field_t *field,
    zvec_db_index_type_t *out_type) {
  if (!field || !field->ptr || !out_type) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  *out_type = static_cast<zvec_db_index_type_t>(field->ptr->index_type());
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_field_get_index_params(
    const zvec_db_field_t *field,
    zvec_db_index_params_t **out_params) {
  if (!field || !field->ptr || !out_params) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  auto params = field->ptr->index_params();
  if (!params) {
    return set_error("index params not set");
  }
  *out_params = new zvec_db_index_params{params->clone()};
  return ZVEC_STATUS_OK;
}

extern "C" int zvec_db_field_is_vector_field(
    const zvec_db_field_t *field) {
  if (!field || !field->ptr) {
    set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return field->ptr->is_vector_field() ? 1 : 0;
}

extern "C" int zvec_db_field_is_dense_vector(
    const zvec_db_field_t *field) {
  if (!field || !field->ptr) {
    set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return field->ptr->is_dense_vector() ? 1 : 0;
}

extern "C" int zvec_db_field_is_sparse_vector(
    const zvec_db_field_t *field) {
  if (!field || !field->ptr) {
    set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return field->ptr->is_sparse_vector() ? 1 : 0;
}

extern "C" int zvec_db_field_is_array_type(const zvec_db_field_t *field) {
  if (!field || !field->ptr) {
    set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return field->ptr->is_array_type() ? 1 : 0;
}

extern "C" zvec_status_t zvec_db_field_to_string(
    const zvec_db_field_t *field,
    char **out_string) {
  if (!field || !field->ptr || !out_string) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  char *dup = dup_string(field->ptr->to_string());
  if (!dup) {
    return set_error("failed to allocate string");
  }
  *out_string = dup;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_field_to_formatted_string(
    const zvec_db_field_t *field,
    int indent_level,
    char **out_string) {
  if (!field || !field->ptr || !out_string) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  char *dup = dup_string(field->ptr->to_string_formatted(indent_level));
  if (!dup) {
    return set_error("failed to allocate string");
  }
  *out_string = dup;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_field_validate(
    const zvec_db_field_t *field) {
  if (!field || !field->ptr) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return status_from_db_status(field->ptr->validate());
}

extern "C" void zvec_db_field_destroy(zvec_db_field_t *field) {
  if (!field) {
    return;
  }
  delete field;
}

extern "C" zvec_status_t zvec_db_index_params_create_invert(
    int enable_range_optimization,
    int enable_extended_wildcard,
    zvec_db_index_params_t **out_params) {
  if (!out_params) {
    return set_error("out_params is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto params = std::make_shared<zvec::InvertIndexParams>(
        enable_range_optimization != 0, enable_extended_wildcard != 0);
    *out_params = new zvec_db_index_params{params};
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_db_index_params_create_hnsw(
    zvec_db_metric_type_t metric,
    int m,
    int ef_construction,
    zvec_db_quantize_type_t quantize,
    zvec_db_index_params_t **out_params) {
  if (!out_params) {
    return set_error("out_params is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto params = std::make_shared<zvec::HnswIndexParams>(
        to_db_metric_type(metric), m, ef_construction,
        to_db_quantize_type(quantize));
    *out_params = new zvec_db_index_params{params};
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_db_index_params_create_ivf(
    zvec_db_metric_type_t metric,
    int n_list,
    int n_iters,
    int use_soar,
    zvec_db_quantize_type_t quantize,
    zvec_db_index_params_t **out_params) {
  if (!out_params) {
    return set_error("out_params is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto params = std::make_shared<zvec::IVFIndexParams>(
        to_db_metric_type(metric), n_list, n_iters, use_soar != 0,
        to_db_quantize_type(quantize));
    *out_params = new zvec_db_index_params{params};
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_db_index_params_create_flat(
    zvec_db_metric_type_t metric,
    zvec_db_quantize_type_t quantize,
    zvec_db_index_params_t **out_params) {
  if (!out_params) {
    return set_error("out_params is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto params = std::make_shared<zvec::FlatIndexParams>(
        to_db_metric_type(metric), to_db_quantize_type(quantize));
    *out_params = new zvec_db_index_params{params};
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_db_index_params_type(
    const zvec_db_index_params_t *params,
    zvec_db_index_type_t *out_type) {
  if (!params || !params->ptr || !out_type) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  *out_type = static_cast<zvec_db_index_type_t>(params->ptr->type());
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_index_params_to_string(
    const zvec_db_index_params_t *params,
    char **out_string) {
  if (!params || !params->ptr || !out_string) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  char *dup = dup_string(params->ptr->to_string());
  if (!dup) {
    return set_error("failed to allocate string");
  }
  *out_string = dup;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_index_params_get_metric(
    const zvec_db_index_params_t *params,
    zvec_db_metric_type_t *out_metric) {
  if (!params || !params->ptr || !out_metric) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  const auto *vec =
      dynamic_cast<const zvec::VectorIndexParams *>(params->ptr.get());
  if (!vec) {
    return set_error("index params does not have metric");
  }
  *out_metric = static_cast<zvec_db_metric_type_t>(vec->metric_type());
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_index_params_get_quantize(
    const zvec_db_index_params_t *params,
    zvec_db_quantize_type_t *out_quantize) {
  if (!params || !params->ptr || !out_quantize) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  const auto *vec =
      dynamic_cast<const zvec::VectorIndexParams *>(params->ptr.get());
  if (!vec) {
    return set_error("index params does not have quantize");
  }
  *out_quantize = static_cast<zvec_db_quantize_type_t>(vec->quantize_type());
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_index_params_get_hnsw_m(
    const zvec_db_index_params_t *params,
    int *out_m) {
  if (!params || !params->ptr || !out_m) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  const auto *hnsw =
      dynamic_cast<const zvec::HnswIndexParams *>(params->ptr.get());
  if (!hnsw) {
    return set_error("index params is not HNSW");
  }
  *out_m = hnsw->m();
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_index_params_get_hnsw_ef_construction(
    const zvec_db_index_params_t *params,
    int *out_ef_construction) {
  if (!params || !params->ptr || !out_ef_construction) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  const auto *hnsw =
      dynamic_cast<const zvec::HnswIndexParams *>(params->ptr.get());
  if (!hnsw) {
    return set_error("index params is not HNSW");
  }
  *out_ef_construction = hnsw->ef_construction();
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_index_params_get_ivf_nlist(
    const zvec_db_index_params_t *params,
    int *out_n_list) {
  if (!params || !params->ptr || !out_n_list) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  const auto *ivf =
      dynamic_cast<const zvec::IVFIndexParams *>(params->ptr.get());
  if (!ivf) {
    return set_error("index params is not IVF");
  }
  *out_n_list = ivf->n_list();
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_index_params_get_ivf_niters(
    const zvec_db_index_params_t *params,
    int *out_n_iters) {
  if (!params || !params->ptr || !out_n_iters) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  const auto *ivf =
      dynamic_cast<const zvec::IVFIndexParams *>(params->ptr.get());
  if (!ivf) {
    return set_error("index params is not IVF");
  }
  *out_n_iters = ivf->n_iters();
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_index_params_get_ivf_use_soar(
    const zvec_db_index_params_t *params,
    int *out_use_soar) {
  if (!params || !params->ptr || !out_use_soar) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  const auto *ivf =
      dynamic_cast<const zvec::IVFIndexParams *>(params->ptr.get());
  if (!ivf) {
    return set_error("index params is not IVF");
  }
  *out_use_soar = ivf->use_soar() ? 1 : 0;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_index_params_get_invert_range_optimization(
    const zvec_db_index_params_t *params,
    int *out_enabled) {
  if (!params || !params->ptr || !out_enabled) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  const auto *invert =
      dynamic_cast<const zvec::InvertIndexParams *>(params->ptr.get());
  if (!invert) {
    return set_error("index params is not invert");
  }
  *out_enabled = invert->enable_range_optimization() ? 1 : 0;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_index_params_get_invert_extended_wildcard(
    const zvec_db_index_params_t *params,
    int *out_enabled) {
  if (!params || !params->ptr || !out_enabled) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  const auto *invert =
      dynamic_cast<const zvec::InvertIndexParams *>(params->ptr.get());
  if (!invert) {
    return set_error("index params is not invert");
  }
  *out_enabled = invert->enable_extended_wildcard() ? 1 : 0;
  return ZVEC_STATUS_OK;
}

extern "C" void zvec_db_index_params_destroy(zvec_db_index_params_t *params) {
  if (!params) {
    return;
  }
  delete params;
}

extern "C" zvec_status_t zvec_db_doc_create(zvec_db_doc_t **out_doc) {
  if (!out_doc) {
    return set_error("out_doc is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto doc = std::make_shared<zvec::Doc>();
    *out_doc = new zvec_db_doc{doc};
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" void zvec_db_doc_destroy(zvec_db_doc_t *doc) {
  if (!doc) {
    return;
  }
  delete doc;
}

extern "C" zvec_status_t zvec_db_doc_set_pk(
    zvec_db_doc_t *doc,
    const char *pk) {
  if (!doc || !doc->ptr || !pk) {
    return set_error("doc or pk is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set_pk(pk);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_doc_id(
    zvec_db_doc_t *doc,
    uint64_t doc_id) {
  if (!doc || !doc->ptr) {
    return set_error("doc is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set_doc_id(doc_id);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_score(
    zvec_db_doc_t *doc,
    float score) {
  if (!doc || !doc->ptr) {
    return set_error("doc is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set_score(score);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_operator(
    zvec_db_doc_t *doc,
    zvec_db_operator_t op) {
  if (!doc || !doc->ptr) {
    return set_error("doc is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set_operator(to_db_operator(op));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_bool(
    zvec_db_doc_t *doc, const char *field, int value) {
  if (!doc || !doc->ptr || !field) {
    return set_error("doc or field is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<bool>(field, value != 0);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_int32(
    zvec_db_doc_t *doc, const char *field, int32_t value) {
  if (!doc || !doc->ptr || !field) {
    return set_error("doc or field is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<int32_t>(field, value);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_int64(
    zvec_db_doc_t *doc, const char *field, int64_t value) {
  if (!doc || !doc->ptr || !field) {
    return set_error("doc or field is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<int64_t>(field, value);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_uint32(
    zvec_db_doc_t *doc, const char *field, uint32_t value) {
  if (!doc || !doc->ptr || !field) {
    return set_error("doc or field is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<uint32_t>(field, value);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_uint64(
    zvec_db_doc_t *doc, const char *field, uint64_t value) {
  if (!doc || !doc->ptr || !field) {
    return set_error("doc or field is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<uint64_t>(field, value);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_float(
    zvec_db_doc_t *doc, const char *field, float value) {
  if (!doc || !doc->ptr || !field) {
    return set_error("doc or field is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<float>(field, value);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_double(
    zvec_db_doc_t *doc, const char *field, double value) {
  if (!doc || !doc->ptr || !field) {
    return set_error("doc or field is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<double>(field, value);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_string(
    zvec_db_doc_t *doc, const char *field, const char *value) {
  if (!doc || !doc->ptr || !field || !value) {
    return set_error("doc, field or value is null",
                     ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<std::string>(field, std::string(value));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_vector_fp32(
    zvec_db_doc_t *doc,
    const char *field,
    const float *values,
    size_t count) {
  if (!doc || !doc->ptr || !field || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<std::vector<float>>(field,
                                    std::vector<float>(values, values + count));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_vector_fp64(
    zvec_db_doc_t *doc,
    const char *field,
    const double *values,
    size_t count) {
  if (!doc || !doc->ptr || !field || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<std::vector<double>>(field,
                                     std::vector<double>(values, values + count));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_vector_int8(
    zvec_db_doc_t *doc,
    const char *field,
    const int8_t *values,
    size_t count) {
  if (!doc || !doc->ptr || !field || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<std::vector<int8_t>>(field,
                                     std::vector<int8_t>(values, values + count));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_vector_int16(
    zvec_db_doc_t *doc,
    const char *field,
    const int16_t *values,
    size_t count) {
  if (!doc || !doc->ptr || !field || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<std::vector<int16_t>>(field,
                                      std::vector<int16_t>(values, values + count));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_vector_int32(
    zvec_db_doc_t *doc,
    const char *field,
    const int32_t *values,
    size_t count) {
  if (!doc || !doc->ptr || !field || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<std::vector<int32_t>>(field,
                                      std::vector<int32_t>(values, values + count));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_vector_int64(
    zvec_db_doc_t *doc,
    const char *field,
    const int64_t *values,
    size_t count) {
  if (!doc || !doc->ptr || !field || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<std::vector<int64_t>>(field,
                                      std::vector<int64_t>(values, values + count));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_vector_uint32(
    zvec_db_doc_t *doc,
    const char *field,
    const uint32_t *values,
    size_t count) {
  if (!doc || !doc->ptr || !field || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<std::vector<uint32_t>>(field,
                                       std::vector<uint32_t>(values, values + count));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_vector_uint64(
    zvec_db_doc_t *doc,
    const char *field,
    const uint64_t *values,
    size_t count) {
  if (!doc || !doc->ptr || !field || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<std::vector<uint64_t>>(field,
                                       std::vector<uint64_t>(values, values + count));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_sparse_vector_fp32(
    zvec_db_doc_t *doc,
    const char *field,
    const uint32_t *indices,
    const float *values,
    size_t count) {
  if (!doc || !doc->ptr || !field || !indices || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  std::vector<uint32_t> idx(indices, indices + count);
  std::vector<float> vals(values, values + count);
  doc->ptr->set<std::pair<std::vector<uint32_t>, std::vector<float>>>(
      field, std::make_pair(std::move(idx), std::move(vals)));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_to_string(
    const zvec_db_doc_t *doc,
    char **out_string) {
  if (!doc || !doc->ptr || !out_string) {
    return set_error("doc or out_string is null",
                     ZVEC_STATUS_INVALID_ARGUMENT);
  }
  char *dup = dup_string(doc->ptr->to_string());
  if (!dup) {
    return set_error("failed to allocate string");
  }
  *out_string = dup;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_to_detail_string(
    const zvec_db_doc_t *doc,
    char **out_string) {
  if (!doc || !doc->ptr || !out_string) {
    return set_error("doc or out_string is null",
                     ZVEC_STATUS_INVALID_ARGUMENT);
  }
  char *dup = dup_string(doc->ptr->to_detail_string());
  if (!dup) {
    return set_error("failed to allocate string");
  }
  *out_string = dup;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_collection_create_and_open(
    const char *path,
    const zvec_db_schema_t *schema,
    int read_only,
    int enable_mmap,
    uint32_t max_buffer_size,
    zvec_db_collection_t **out_collection) {
  if (!path || !schema || !schema->ptr || !out_collection) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    zvec::CollectionOptions options(read_only != 0, enable_mmap != 0,
                                    max_buffer_size);
    auto result =
        zvec::Collection::CreateAndOpen(path, *schema->ptr, options);
    if (!result.has_value()) {
      return set_error(result.error().message());
    }
    *out_collection = new zvec_db_collection{result.value()};
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_db_collection_open(
    const char *path,
    int read_only,
    int enable_mmap,
    uint32_t max_buffer_size,
    zvec_db_collection_t **out_collection) {
  if (!path || !out_collection) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    zvec::CollectionOptions options(read_only != 0, enable_mmap != 0,
                                    max_buffer_size);
    auto result = zvec::Collection::Open(path, options);
    if (!result.has_value()) {
      return set_error(result.error().message());
    }
    *out_collection = new zvec_db_collection{result.value()};
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" void zvec_db_collection_destroy(
    zvec_db_collection_t *collection) {
  if (!collection) {
    return;
  }
  delete collection;
}

extern "C" zvec_status_t zvec_db_collection_flush(
    zvec_db_collection_t *collection) {
  if (!collection || !collection->ptr) {
    return set_error("collection is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return status_from_db_status(collection->ptr->Flush());
}

extern "C" zvec_status_t zvec_db_collection_optimize(
    zvec_db_collection_t *collection) {
  if (!collection || !collection->ptr) {
    return set_error("collection is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return status_from_db_status(collection->ptr->Optimize());
}

extern "C" zvec_status_t zvec_db_collection_stats(
    zvec_db_collection_t *collection,
    char **out_string) {
  if (!collection || !collection->ptr || !out_string) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  auto result = collection->ptr->Stats();
  if (!result.has_value()) {
    return set_error(result.error().message());
  }
  char *dup = dup_string(result.value().to_string());
  if (!dup) {
    return set_error("failed to allocate string");
  }
  *out_string = dup;
  return ZVEC_STATUS_OK;
}

static zvec_status_t collection_write_common(
    zvec::Collection::Ptr &collection,
    zvec_db_doc_t **docs,
    size_t count,
    const char *op_name,
    const std::function<zvec::Result<zvec::WriteResults>(
        std::vector<zvec::Doc> &)> &fn) {
  if (!collection || !docs || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  std::vector<zvec::Doc> input;
  input.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    if (!docs[i] || !docs[i]->ptr) {
      return set_error("doc is null", ZVEC_STATUS_INVALID_ARGUMENT);
    }
    input.push_back(*docs[i]->ptr);
  }
  auto result = fn(input);
  if (!result.has_value()) {
    return set_error(result.error().message());
  }
  for (const auto &s : result.value()) {
    if (!s.ok()) {
      return set_error(op_name + std::string(" failed: ") + s.message());
    }
  }
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_collection_insert(
    zvec_db_collection_t *collection,
    zvec_db_doc_t **docs,
    size_t count) {
  if (!collection || !collection->ptr) {
    return set_error("collection is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return collection_write_common(
      collection->ptr, docs, count, "insert",
      [&](std::vector<zvec::Doc> &input) {
        return collection->ptr->Insert(input);
      });
}

extern "C" zvec_status_t zvec_db_collection_upsert(
    zvec_db_collection_t *collection,
    zvec_db_doc_t **docs,
    size_t count) {
  if (!collection || !collection->ptr) {
    return set_error("collection is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return collection_write_common(
      collection->ptr, docs, count, "upsert",
      [&](std::vector<zvec::Doc> &input) {
        return collection->ptr->Upsert(input);
      });
}

extern "C" zvec_status_t zvec_db_collection_update(
    zvec_db_collection_t *collection,
    zvec_db_doc_t **docs,
    size_t count) {
  if (!collection || !collection->ptr) {
    return set_error("collection is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return collection_write_common(
      collection->ptr, docs, count, "update",
      [&](std::vector<zvec::Doc> &input) {
        return collection->ptr->Update(input);
      });
}

extern "C" zvec_status_t zvec_db_query_params_create_hnsw(
    int ef,
    float radius,
    int is_linear,
    int is_using_refiner,
    zvec_db_query_params_t **out_params) {
  if (!out_params) {
    return set_error("out_params is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto qp = std::make_shared<zvec::HnswQueryParams>(
        ef, radius, is_linear != 0, is_using_refiner != 0);
    *out_params = new zvec_db_query_params{qp};
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_db_query_params_create_ivf(
    int nprobe,
    float scale_factor,
    int is_using_refiner,
    zvec_db_query_params_t **out_params) {
  if (!out_params) {
    return set_error("out_params is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto qp = std::make_shared<zvec::IVFQueryParams>(
        nprobe, is_using_refiner != 0, scale_factor);
    *out_params = new zvec_db_query_params{qp};
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_db_query_params_create_flat(
    float scale_factor,
    int is_using_refiner,
    zvec_db_query_params_t **out_params) {
  if (!out_params) {
    return set_error("out_params is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto qp = std::make_shared<zvec::FlatQueryParams>(
        is_using_refiner != 0, scale_factor);
    *out_params = new zvec_db_query_params{qp};
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" zvec_status_t zvec_db_query_params_set_radius(
    zvec_db_query_params_t *params, float radius) {
  if (!params || !params->ptr) {
    return set_error("params is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  params->ptr->set_radius(radius);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_query_params_get_radius(
    const zvec_db_query_params_t *params, float *out_radius) {
  if (!params || !params->ptr || !out_radius) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  *out_radius = params->ptr->radius();
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_query_params_set_is_linear(
    zvec_db_query_params_t *params, int is_linear) {
  if (!params || !params->ptr) {
    return set_error("params is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  params->ptr->set_is_linear(is_linear != 0);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_query_params_get_is_linear(
    const zvec_db_query_params_t *params, int *out_is_linear) {
  if (!params || !params->ptr || !out_is_linear) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  *out_is_linear = params->ptr->is_linear() ? 1 : 0;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_query_params_set_is_using_refiner(
    zvec_db_query_params_t *params, int is_using_refiner) {
  if (!params || !params->ptr) {
    return set_error("params is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  params->ptr->set_is_using_refiner(is_using_refiner != 0);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_query_params_get_is_using_refiner(
    const zvec_db_query_params_t *params, int *out_is_using_refiner) {
  if (!params || !params->ptr || !out_is_using_refiner) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  *out_is_using_refiner = params->ptr->is_using_refiner() ? 1 : 0;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_query_params_set_hnsw_ef(
    zvec_db_query_params_t *params, int ef) {
  if (!params || !params->ptr) {
    return set_error("params is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  auto *hnsw = dynamic_cast<zvec::HnswQueryParams *>(params->ptr.get());
  if (!hnsw) {
    return set_error("query params is not HNSW");
  }
  hnsw->set_ef(ef);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_query_params_get_hnsw_ef(
    const zvec_db_query_params_t *params, int *out_ef) {
  if (!params || !params->ptr || !out_ef) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  const auto *hnsw =
      dynamic_cast<const zvec::HnswQueryParams *>(params->ptr.get());
  if (!hnsw) {
    return set_error("query params is not HNSW");
  }
  *out_ef = hnsw->ef();
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_query_params_set_ivf_nprobe(
    zvec_db_query_params_t *params, int nprobe) {
  if (!params || !params->ptr) {
    return set_error("params is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  auto *ivf = dynamic_cast<zvec::IVFQueryParams *>(params->ptr.get());
  if (!ivf) {
    return set_error("query params is not IVF");
  }
  ivf->set_nprobe(nprobe);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_query_params_get_ivf_nprobe(
    const zvec_db_query_params_t *params, int *out_nprobe) {
  if (!params || !params->ptr || !out_nprobe) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  const auto *ivf =
      dynamic_cast<const zvec::IVFQueryParams *>(params->ptr.get());
  if (!ivf) {
    return set_error("query params is not IVF");
  }
  *out_nprobe = ivf->nprobe();
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_query_params_set_scale_factor(
    zvec_db_query_params_t *params, float scale_factor) {
  if (!params || !params->ptr) {
    return set_error("params is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  if (auto *ivf = dynamic_cast<zvec::IVFQueryParams *>(params->ptr.get())) {
    ivf->set_scale_factor(scale_factor);
    return ZVEC_STATUS_OK;
  }
  if (auto *flat = dynamic_cast<zvec::FlatQueryParams *>(params->ptr.get())) {
    flat->set_scale_factor(scale_factor);
    return ZVEC_STATUS_OK;
  }
  return set_error("query params does not support scale factor");
}

extern "C" zvec_status_t zvec_db_query_params_get_scale_factor(
    const zvec_db_query_params_t *params, float *out_scale_factor) {
  if (!params || !params->ptr || !out_scale_factor) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  if (const auto *ivf =
          dynamic_cast<const zvec::IVFQueryParams *>(params->ptr.get())) {
    *out_scale_factor = ivf->scale_factor();
    return ZVEC_STATUS_OK;
  }
  if (const auto *flat =
          dynamic_cast<const zvec::FlatQueryParams *>(params->ptr.get())) {
    *out_scale_factor = flat->scale_factor();
    return ZVEC_STATUS_OK;
  }
  return set_error("query params does not support scale factor");
}

extern "C" void zvec_db_query_params_destroy(
    zvec_db_query_params_t *params) {
  if (!params) {
    return;
  }
  delete params;
}

extern "C" zvec_status_t zvec_db_query_create(
    const char *field_name,
    zvec_db_query_t **out_query) {
  if (!field_name || !out_query) {
    return set_error("field_name or out_query is null",
                     ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    auto *q = new zvec_db_query();
    q->query.field_name_ = field_name;
    *out_query = q;
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" void zvec_db_query_destroy(zvec_db_query_t *query) {
  if (!query) {
    return;
  }
  delete query;
}

extern "C" zvec_status_t zvec_db_query_set_topk(
    zvec_db_query_t *query, int topk) {
  if (!query) {
    return set_error("query is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  query->query.topk_ = topk;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_query_set_include_vector(
    zvec_db_query_t *query, int include_vector) {
  if (!query) {
    return set_error("query is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  query->query.include_vector_ = include_vector != 0;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_query_set_include_doc_id(
    zvec_db_query_t *query, int include_doc_id) {
  if (!query) {
    return set_error("query is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  query->query.include_doc_id_ = include_doc_id != 0;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_query_set_filter(
    zvec_db_query_t *query, const char *filter) {
  if (!query || !filter) {
    return set_error("query or filter is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  query->query.filter_ = filter;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_query_set_output_fields(
    zvec_db_query_t *query, const char **fields, size_t count) {
  if (!query) {
    return set_error("query is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  if (!fields) {
    query->query.output_fields_.reset();
    return ZVEC_STATUS_OK;
  }
  std::vector<std::string> out;
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    if (fields[i]) {
      out.emplace_back(fields[i]);
    }
  }
  query->query.output_fields_ = std::move(out);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_query_set_dense_vector_bytes(
    zvec_db_query_t *query, const void *data, size_t bytes) {
  if (!query || !data || bytes == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  query->query.query_vector_.assign(
      reinterpret_cast<const char *>(data), bytes);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_query_set_sparse_vector(
    zvec_db_query_t *query,
    const uint32_t *indices,
    const void *values,
    size_t count,
    size_t value_bytes) {
  if (!query || !indices || !values || count == 0 || value_bytes == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  query->query.query_sparse_indices_.assign(
      reinterpret_cast<const char *>(indices),
      count * sizeof(uint32_t));
  query->query.query_sparse_values_.assign(
      reinterpret_cast<const char *>(values), value_bytes);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_query_set_params(
    zvec_db_query_t *query,
    const zvec_db_query_params_t *params) {
  if (!query || !params || !params->ptr) {
    return set_error("query or params is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  query->query.query_params_ = params->ptr;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_collection_query(
    zvec_db_collection_t *collection,
    const zvec_db_query_t *query,
    zvec_db_query_result_t **out_result) {
  if (!collection || !collection->ptr || !query || !out_result) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  auto result = collection->ptr->Query(query->query);
  if (!result.has_value()) {
    return set_error(result.error().message());
  }
  auto *wrap = new zvec_db_query_result;
  wrap->docs = result.value();
  *out_result = wrap;
  return ZVEC_STATUS_OK;
}

extern "C" size_t zvec_db_query_result_size(
    const zvec_db_query_result_t *result) {
  if (!result) {
    set_error("result is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return result->docs.size();
}

extern "C" zvec_status_t zvec_db_query_result_get_doc(
    const zvec_db_query_result_t *result,
    size_t idx,
    zvec_db_doc_t **out_doc) {
  if (!result || !out_doc) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  if (idx >= result->docs.size()) {
    return set_error("index out of range", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  auto doc = result->docs[idx];
  if (!doc) {
    return set_error("doc is null");
  }
  *out_doc = new zvec_db_doc{doc};
  return ZVEC_STATUS_OK;
}

extern "C" void zvec_db_query_result_destroy(
    zvec_db_query_result_t *result) {
  if (!result) {
    return;
  }
  delete result;
}

extern "C" zvec_status_t zvec_db_doc_set_binary(
    zvec_db_doc_t *doc, const char *field, const void *data, size_t size) {
  if (!doc || !doc->ptr || !field || !data || size == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<std::string>(
      field, std::string(reinterpret_cast<const char *>(data), size));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_null(
    zvec_db_doc_t *doc, const char *field) {
  if (!doc || !doc->ptr || !field) {
    return set_error("doc or field is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set_null(field);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_remove_field(
    zvec_db_doc_t *doc, const char *field) {
  if (!doc || !doc->ptr || !field) {
    return set_error("doc or field is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->remove(field);
  return ZVEC_STATUS_OK;
}

extern "C" void zvec_db_doc_clear(zvec_db_doc_t *doc) {
  if (!doc || !doc->ptr) {
    set_error("doc is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return;
  }
  doc->ptr->clear();
}

extern "C" zvec_status_t zvec_db_doc_set_vector_fp16(
    zvec_db_doc_t *doc,
    const char *field,
    const float *values,
    size_t count) {
  if (!doc || !doc->ptr || !field || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  std::vector<zvec::ailego::Float16> out;
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    out.emplace_back(values[i]);
  }
  doc->ptr->set<std::vector<zvec::ailego::Float16>>(field, std::move(out));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_vector_binary32(
    zvec_db_doc_t *doc,
    const char *field,
    const uint32_t *values,
    size_t count) {
  return zvec_db_doc_set_vector_uint32(doc, field, values, count);
}

extern "C" zvec_status_t zvec_db_doc_set_vector_binary64(
    zvec_db_doc_t *doc,
    const char *field,
    const uint64_t *values,
    size_t count) {
  return zvec_db_doc_set_vector_uint64(doc, field, values, count);
}

extern "C" zvec_status_t zvec_db_doc_set_vector_int4(
    zvec_db_doc_t *doc,
    const char *field,
    const int8_t *values,
    size_t count) {
  return zvec_db_doc_set_vector_int8(doc, field, values, count);
}

extern "C" zvec_status_t zvec_db_doc_set_sparse_vector_fp16(
    zvec_db_doc_t *doc,
    const char *field,
    const uint32_t *indices,
    const float *values,
    size_t count) {
  if (!doc || !doc->ptr || !field || !indices || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  std::vector<uint32_t> idx(indices, indices + count);
  std::vector<zvec::ailego::Float16> vals;
  vals.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    vals.emplace_back(values[i]);
  }
  doc->ptr->set<std::pair<std::vector<uint32_t>, std::vector<zvec::ailego::Float16>>>(
      field, std::make_pair(std::move(idx), std::move(vals)));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_array_bool(
    zvec_db_doc_t *doc, const char *field,
    const uint8_t *values, size_t count) {
  if (!doc || !doc->ptr || !field || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  std::vector<bool> out;
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    out.push_back(values[i] != 0);
  }
  doc->ptr->set<std::vector<bool>>(field, std::move(out));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_array_int32(
    zvec_db_doc_t *doc, const char *field,
    const int32_t *values, size_t count) {
  if (!doc || !doc->ptr || !field || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<std::vector<int32_t>>(
      field, std::vector<int32_t>(values, values + count));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_array_int64(
    zvec_db_doc_t *doc, const char *field,
    const int64_t *values, size_t count) {
  if (!doc || !doc->ptr || !field || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<std::vector<int64_t>>(
      field, std::vector<int64_t>(values, values + count));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_array_uint32(
    zvec_db_doc_t *doc, const char *field,
    const uint32_t *values, size_t count) {
  if (!doc || !doc->ptr || !field || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<std::vector<uint32_t>>(
      field, std::vector<uint32_t>(values, values + count));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_array_uint64(
    zvec_db_doc_t *doc, const char *field,
    const uint64_t *values, size_t count) {
  if (!doc || !doc->ptr || !field || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<std::vector<uint64_t>>(
      field, std::vector<uint64_t>(values, values + count));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_array_float(
    zvec_db_doc_t *doc, const char *field,
    const float *values, size_t count) {
  if (!doc || !doc->ptr || !field || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<std::vector<float>>(
      field, std::vector<float>(values, values + count));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_array_double(
    zvec_db_doc_t *doc, const char *field,
    const double *values, size_t count) {
  if (!doc || !doc->ptr || !field || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  doc->ptr->set<std::vector<double>>(
      field, std::vector<double>(values, values + count));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_array_string(
    zvec_db_doc_t *doc, const char *field,
    const char **values, size_t count) {
  if (!doc || !doc->ptr || !field || !values || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  std::vector<std::string> out;
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    out.emplace_back(values[i] ? values[i] : "");
  }
  doc->ptr->set<std::vector<std::string>>(field, std::move(out));
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_set_array_binary(
    zvec_db_doc_t *doc, const char *field,
    const char **values, size_t count) {
  return zvec_db_doc_set_array_string(doc, field, values, count);
}

extern "C" zvec_status_t zvec_db_doc_get_pk(
    const zvec_db_doc_t *doc, char **out_pk) {
  if (!doc || !doc->ptr || !out_pk) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  char *dup = dup_string(doc->ptr->pk());
  if (!dup) {
    return set_error("failed to allocate string");
  }
  *out_pk = dup;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_get_doc_id(
    const zvec_db_doc_t *doc, uint64_t *out_doc_id) {
  if (!doc || !doc->ptr || !out_doc_id) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  *out_doc_id = doc->ptr->doc_id();
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_get_score(
    const zvec_db_doc_t *doc, float *out_score) {
  if (!doc || !doc->ptr || !out_score) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  *out_score = doc->ptr->score();
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_get_operator(
    const zvec_db_doc_t *doc, zvec_db_operator_t *out_op) {
  if (!doc || !doc->ptr || !out_op) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  *out_op = static_cast<zvec_db_operator_t>(doc->ptr->get_operator());
  return ZVEC_STATUS_OK;
}

extern "C" int zvec_db_doc_has(const zvec_db_doc_t *doc, const char *field) {
  if (!doc || !doc->ptr || !field) {
    set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return doc->ptr->has(field) ? 1 : 0;
}

extern "C" int zvec_db_doc_has_value(
    const zvec_db_doc_t *doc, const char *field) {
  if (!doc || !doc->ptr || !field) {
    set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return doc->ptr->has_value(field) ? 1 : 0;
}

extern "C" int zvec_db_doc_is_null(
    const zvec_db_doc_t *doc, const char *field) {
  if (!doc || !doc->ptr || !field) {
    set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return doc->ptr->is_null(field) ? 1 : 0;
}

extern "C" int zvec_db_doc_is_empty(const zvec_db_doc_t *doc) {
  if (!doc || !doc->ptr) {
    set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return doc->ptr->is_empty() ? 1 : 0;
}

extern "C" zvec_status_t zvec_db_doc_field_names(
    const zvec_db_doc_t *doc,
    char ***out_names,
    size_t *out_count) {
  if (!doc || !doc->ptr || !out_names || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  auto names = doc->ptr->field_names();
  return copy_string_vector_out(names, out_names, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_bool(
    const zvec_db_doc_t *doc, const char *field, int *out_value) {
  bool value = false;
  auto status = get_field_value<bool>(doc, field, "bool", &value);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  *out_value = value ? 1 : 0;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_get_int32(
    const zvec_db_doc_t *doc, const char *field, int32_t *out_value) {
  return get_field_value<int32_t>(doc, field, "int32", out_value);
}

extern "C" zvec_status_t zvec_db_doc_get_int64(
    const zvec_db_doc_t *doc, const char *field, int64_t *out_value) {
  return get_field_value<int64_t>(doc, field, "int64", out_value);
}

extern "C" zvec_status_t zvec_db_doc_get_uint32(
    const zvec_db_doc_t *doc, const char *field, uint32_t *out_value) {
  return get_field_value<uint32_t>(doc, field, "uint32", out_value);
}

extern "C" zvec_status_t zvec_db_doc_get_uint64(
    const zvec_db_doc_t *doc, const char *field, uint64_t *out_value) {
  return get_field_value<uint64_t>(doc, field, "uint64", out_value);
}

extern "C" zvec_status_t zvec_db_doc_get_float(
    const zvec_db_doc_t *doc, const char *field, float *out_value) {
  return get_field_value<float>(doc, field, "float", out_value);
}

extern "C" zvec_status_t zvec_db_doc_get_double(
    const zvec_db_doc_t *doc, const char *field, double *out_value) {
  return get_field_value<double>(doc, field, "double", out_value);
}

extern "C" zvec_status_t zvec_db_doc_get_string(
    const zvec_db_doc_t *doc, const char *field, char **out_string) {
  std::string value;
  auto status = get_field_value<std::string>(doc, field, "string", &value);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  char *dup = dup_string(value);
  if (!dup) {
    return set_error("failed to allocate string");
  }
  *out_string = dup;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_get_binary(
    const zvec_db_doc_t *doc, const char *field,
    void **out_data, size_t *out_size) {
  if (!out_data || !out_size) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  std::string value;
  auto status = get_field_value<std::string>(doc, field, "binary", &value);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  if (value.empty()) {
    *out_data = nullptr;
    *out_size = 0;
    return ZVEC_STATUS_OK;
  }
  void *buf = std::malloc(value.size());
  if (!buf) {
    return set_error("failed to allocate binary");
  }
  std::memcpy(buf, value.data(), value.size());
  *out_data = buf;
  *out_size = value.size();
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_get_vector_fp32(
    const zvec_db_doc_t *doc, const char *field,
    float **out_values, size_t *out_count) {
  std::vector<float> values;
  auto status = get_field_value<std::vector<float>>(
      doc, field, "vector<float>", &values);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  return copy_vector_out(values, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_vector_fp64(
    const zvec_db_doc_t *doc, const char *field,
    double **out_values, size_t *out_count) {
  std::vector<double> values;
  auto status = get_field_value<std::vector<double>>(
      doc, field, "vector<double>", &values);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  return copy_vector_out(values, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_vector_fp16(
    const zvec_db_doc_t *doc, const char *field,
    float **out_values, size_t *out_count) {
  std::vector<zvec::ailego::Float16> values;
  auto status = get_field_value<std::vector<zvec::ailego::Float16>>(
      doc, field, "vector<float16>", &values);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  return copy_fp16_vector_out(values, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_vector_int8(
    const zvec_db_doc_t *doc, const char *field,
    int8_t **out_values, size_t *out_count) {
  std::vector<int8_t> values;
  auto status = get_field_value<std::vector<int8_t>>(
      doc, field, "vector<int8>", &values);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  return copy_vector_out(values, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_vector_int16(
    const zvec_db_doc_t *doc, const char *field,
    int16_t **out_values, size_t *out_count) {
  std::vector<int16_t> values;
  auto status = get_field_value<std::vector<int16_t>>(
      doc, field, "vector<int16>", &values);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  return copy_vector_out(values, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_vector_int32(
    const zvec_db_doc_t *doc, const char *field,
    int32_t **out_values, size_t *out_count) {
  std::vector<int32_t> values;
  auto status = get_field_value<std::vector<int32_t>>(
      doc, field, "vector<int32>", &values);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  return copy_vector_out(values, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_vector_int64(
    const zvec_db_doc_t *doc, const char *field,
    int64_t **out_values, size_t *out_count) {
  std::vector<int64_t> values;
  auto status = get_field_value<std::vector<int64_t>>(
      doc, field, "vector<int64>", &values);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  return copy_vector_out(values, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_vector_uint32(
    const zvec_db_doc_t *doc, const char *field,
    uint32_t **out_values, size_t *out_count) {
  std::vector<uint32_t> values;
  auto status = get_field_value<std::vector<uint32_t>>(
      doc, field, "vector<uint32>", &values);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  return copy_vector_out(values, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_vector_uint64(
    const zvec_db_doc_t *doc, const char *field,
    uint64_t **out_values, size_t *out_count) {
  std::vector<uint64_t> values;
  auto status = get_field_value<std::vector<uint64_t>>(
      doc, field, "vector<uint64>", &values);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  return copy_vector_out(values, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_vector_binary32(
    const zvec_db_doc_t *doc, const char *field,
    uint32_t **out_values, size_t *out_count) {
  return zvec_db_doc_get_vector_uint32(doc, field, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_vector_binary64(
    const zvec_db_doc_t *doc, const char *field,
    uint64_t **out_values, size_t *out_count) {
  return zvec_db_doc_get_vector_uint64(doc, field, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_vector_int4(
    const zvec_db_doc_t *doc, const char *field,
    int8_t **out_values, size_t *out_count) {
  return zvec_db_doc_get_vector_int8(doc, field, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_sparse_vector_fp32(
    const zvec_db_doc_t *doc, const char *field,
    uint32_t **out_indices,
    float **out_values,
    size_t *out_count) {
  if (!out_indices || !out_values || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  std::pair<std::vector<uint32_t>, std::vector<float>> value;
  auto status =
      get_field_value<std::pair<std::vector<uint32_t>, std::vector<float>>>(
          doc, field, "sparse<float>", &value);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  if (value.first.size() != value.second.size()) {
    return set_error("sparse vector size mismatch");
  }
  auto s1 = copy_vector_out(value.first, out_indices, out_count);
  if (s1 != ZVEC_STATUS_OK) {
    return s1;
  }
  size_t val_count = 0;
  auto s2 = copy_vector_out(value.second, out_values, &val_count);
  if (s2 != ZVEC_STATUS_OK) {
    std::free(*out_indices);
    *out_indices = nullptr;
    *out_count = 0;
    return s2;
  }
  if (val_count != *out_count) {
    std::free(*out_indices);
    std::free(*out_values);
    *out_indices = nullptr;
    *out_values = nullptr;
    *out_count = 0;
    return set_error("sparse vector size mismatch");
  }
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_get_sparse_vector_fp16(
    const zvec_db_doc_t *doc, const char *field,
    uint32_t **out_indices,
    float **out_values,
    size_t *out_count) {
  if (!out_indices || !out_values || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  std::pair<std::vector<uint32_t>, std::vector<zvec::ailego::Float16>> value;
  auto status =
      get_field_value<std::pair<std::vector<uint32_t>,
                                std::vector<zvec::ailego::Float16>>>(
          doc, field, "sparse<float16>", &value);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  if (value.first.size() != value.second.size()) {
    return set_error("sparse vector size mismatch");
  }
  auto s1 = copy_vector_out(value.first, out_indices, out_count);
  if (s1 != ZVEC_STATUS_OK) {
    return s1;
  }
  if (value.second.empty()) {
    *out_values = nullptr;
    return ZVEC_STATUS_OK;
  }
  float *vals =
      static_cast<float *>(std::malloc(value.second.size() * sizeof(float)));
  if (!vals) {
    std::free(*out_indices);
    *out_indices = nullptr;
    *out_count = 0;
    return set_error("failed to allocate output");
  }
  for (size_t i = 0; i < value.second.size(); ++i) {
    vals[i] = static_cast<float>(value.second[i]);
  }
  *out_values = vals;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_doc_get_array_bool(
    const zvec_db_doc_t *doc, const char *field,
    uint8_t **out_values, size_t *out_count) {
  std::vector<bool> values;
  auto status = get_field_value<std::vector<bool>>(
      doc, field, "array<bool>", &values);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  return copy_bool_vector_out(values, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_array_int32(
    const zvec_db_doc_t *doc, const char *field,
    int32_t **out_values, size_t *out_count) {
  std::vector<int32_t> values;
  auto status = get_field_value<std::vector<int32_t>>(
      doc, field, "array<int32>", &values);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  return copy_vector_out(values, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_array_int64(
    const zvec_db_doc_t *doc, const char *field,
    int64_t **out_values, size_t *out_count) {
  std::vector<int64_t> values;
  auto status = get_field_value<std::vector<int64_t>>(
      doc, field, "array<int64>", &values);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  return copy_vector_out(values, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_array_uint32(
    const zvec_db_doc_t *doc, const char *field,
    uint32_t **out_values, size_t *out_count) {
  std::vector<uint32_t> values;
  auto status = get_field_value<std::vector<uint32_t>>(
      doc, field, "array<uint32>", &values);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  return copy_vector_out(values, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_array_uint64(
    const zvec_db_doc_t *doc, const char *field,
    uint64_t **out_values, size_t *out_count) {
  std::vector<uint64_t> values;
  auto status = get_field_value<std::vector<uint64_t>>(
      doc, field, "array<uint64>", &values);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  return copy_vector_out(values, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_array_float(
    const zvec_db_doc_t *doc, const char *field,
    float **out_values, size_t *out_count) {
  std::vector<float> values;
  auto status = get_field_value<std::vector<float>>(
      doc, field, "array<float>", &values);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  return copy_vector_out(values, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_array_double(
    const zvec_db_doc_t *doc, const char *field,
    double **out_values, size_t *out_count) {
  std::vector<double> values;
  auto status = get_field_value<std::vector<double>>(
      doc, field, "array<double>", &values);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  return copy_vector_out(values, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_array_string(
    const zvec_db_doc_t *doc, const char *field,
    char ***out_values, size_t *out_count) {
  std::vector<std::string> values;
  auto status = get_field_value<std::vector<std::string>>(
      doc, field, "array<string>", &values);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  return copy_string_vector_out(values, out_values, out_count);
}

extern "C" zvec_status_t zvec_db_doc_get_array_binary(
    const zvec_db_doc_t *doc, const char *field,
    void ***out_values, size_t **out_sizes, size_t *out_count) {
  std::vector<std::string> values;
  auto status = get_field_value<std::vector<std::string>>(
      doc, field, "array<binary>", &values);
  if (status != ZVEC_STATUS_OK) {
    return status;
  }
  return copy_binary_vector_out(values, out_values, out_sizes, out_count);
}

extern "C" size_t zvec_db_doc_memory_usage(const zvec_db_doc_t *doc) {
  if (!doc || !doc->ptr) {
    set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return doc->ptr->memory_usage();
}

extern "C" zvec_status_t zvec_db_doc_validate(
    const zvec_db_doc_t *doc,
    const zvec_db_schema_t *schema,
    int is_update) {
  if (!doc || !doc->ptr || !schema || !schema->ptr) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return status_from_db_status(
      doc->ptr->validate(schema->ptr, is_update != 0));
}

extern "C" void zvec_db_string_array_free(char **items, size_t count) {
  if (!items) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    std::free(items[i]);
  }
  std::free(items);
}

extern "C" void zvec_db_binary_array_free(
    void **items, size_t *sizes, size_t count) {
  if (items) {
    for (size_t i = 0; i < count; ++i) {
      std::free(items[i]);
    }
    std::free(items);
  }
  std::free(sizes);
}

extern "C" zvec_status_t zvec_db_collection_destroy_storage(
    zvec_db_collection_t *collection) {
  if (!collection || !collection->ptr) {
    return set_error("collection is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return status_from_db_status(collection->ptr->Destroy());
}

extern "C" zvec_status_t zvec_db_collection_path(
    zvec_db_collection_t *collection,
    char **out_string) {
  if (!collection || !collection->ptr || !out_string) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  auto result = collection->ptr->Path();
  if (!result.has_value()) {
    return set_error(result.error().message());
  }
  char *dup = dup_string(result.value());
  if (!dup) {
    return set_error("failed to allocate string");
  }
  *out_string = dup;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_collection_get_schema(
    zvec_db_collection_t *collection,
    zvec_db_schema_t **out_schema) {
  if (!collection || !collection->ptr || !out_schema) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  auto result = collection->ptr->Schema();
  if (!result.has_value()) {
    return set_error(result.error().message());
  }
  auto schema = std::make_shared<zvec::CollectionSchema>(result.value());
  *out_schema = new zvec_db_schema{schema};
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_collection_get_options(
    zvec_db_collection_t *collection,
    int *out_read_only,
    int *out_enable_mmap,
    uint32_t *out_max_buffer_size) {
  if (!collection || !collection->ptr || !out_read_only || !out_enable_mmap ||
      !out_max_buffer_size) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  auto result = collection->ptr->Options();
  if (!result.has_value()) {
    return set_error(result.error().message());
  }
  const auto &opt = result.value();
  *out_read_only = opt.read_only_ ? 1 : 0;
  *out_enable_mmap = opt.enable_mmap_ ? 1 : 0;
  *out_max_buffer_size = opt.max_buffer_size_;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_collection_create_index(
    zvec_db_collection_t *collection,
    const char *field,
    const zvec_db_index_params_t *params,
    int concurrency) {
  if (!collection || !collection->ptr || !field || !params || !params->ptr) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  zvec::CreateIndexOptions options;
  options.concurrency_ = concurrency;
  return status_from_db_status(
      collection->ptr->CreateIndex(field, params->ptr, options));
}

extern "C" zvec_status_t zvec_db_collection_drop_index(
    zvec_db_collection_t *collection,
    const char *field) {
  if (!collection || !collection->ptr || !field) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return status_from_db_status(collection->ptr->DropIndex(field));
}

extern "C" zvec_status_t zvec_db_collection_add_column(
    zvec_db_collection_t *collection,
    const zvec_db_field_t *field,
    const char *expression,
    int concurrency) {
  if (!collection || !collection->ptr || !field || !field->ptr) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  zvec::AddColumnOptions options;
  options.concurrency_ = concurrency;
  std::string expr = expression ? expression : "";
  return status_from_db_status(
      collection->ptr->AddColumn(field->ptr, expr, options));
}

extern "C" zvec_status_t zvec_db_collection_drop_column(
    zvec_db_collection_t *collection,
    const char *field) {
  if (!collection || !collection->ptr || !field) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return status_from_db_status(collection->ptr->DropColumn(field));
}

extern "C" zvec_status_t zvec_db_collection_alter_column(
    zvec_db_collection_t *collection,
    const char *field,
    const char *rename,
    const zvec_db_field_t *new_field,
    int concurrency) {
  if (!collection || !collection->ptr || !field) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  zvec::AlterColumnOptions options;
  options.concurrency_ = concurrency;
  std::string rename_str = rename ? rename : "";
  zvec::FieldSchema::Ptr new_schema = new_field ? new_field->ptr : nullptr;
  return status_from_db_status(
      collection->ptr->AlterColumn(field, rename_str, new_schema, options));
}

extern "C" zvec_status_t zvec_db_collection_delete(
    zvec_db_collection_t *collection,
    const char **pks,
    size_t count) {
  if (!collection || !collection->ptr || !pks || count == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  std::vector<std::string> keys;
  keys.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    keys.emplace_back(pks[i] ? pks[i] : "");
  }
  auto result = collection->ptr->Delete(keys);
  if (!result.has_value()) {
    return set_error(result.error().message());
  }
  for (const auto &s : result.value()) {
    if (!s.ok()) {
      return set_error(std::string("delete failed: ") + s.message());
    }
  }
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_collection_delete_by_filter(
    zvec_db_collection_t *collection,
    const char *filter) {
  if (!collection || !collection->ptr || !filter) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return status_from_db_status(collection->ptr->DeleteByFilter(filter));
}

extern "C" zvec_status_t zvec_db_collection_fetch(
    zvec_db_collection_t *collection,
    const char **pks,
    size_t count,
    zvec_db_query_result_t **out_result) {
  if (!collection || !collection->ptr || !pks || count == 0 || !out_result) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  std::vector<std::string> keys;
  keys.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    keys.emplace_back(pks[i] ? pks[i] : "");
  }
  auto result = collection->ptr->Fetch(keys);
  if (!result.has_value()) {
    return set_error(result.error().message());
  }
  auto *wrap = new zvec_db_query_result;
  wrap->docs.reserve(count);
  const auto &map = result.value();
  for (const auto &k : keys) {
    auto it = map.find(k);
    if (it != map.end()) {
      wrap->docs.push_back(it->second);
    }
  }
  *out_result = wrap;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_group_query_create(
    const char *field_name,
    const char *group_by_field,
    zvec_db_group_query_t **out_query) {
  if (!field_name || !group_by_field || !out_query) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  auto *q = new zvec_db_group_query();
  q->query.field_name_ = field_name;
  q->query.group_by_field_name_ = group_by_field;
  *out_query = q;
  return ZVEC_STATUS_OK;
}

extern "C" void zvec_db_group_query_destroy(zvec_db_group_query_t *query) {
  if (!query) {
    return;
  }
  delete query;
}

extern "C" zvec_status_t zvec_db_group_query_set_group_count(
    zvec_db_group_query_t *query, uint32_t group_count) {
  if (!query) {
    return set_error("query is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  query->query.group_count_ = group_count;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_group_query_set_group_topk(
    zvec_db_group_query_t *query, uint32_t group_topk) {
  if (!query) {
    return set_error("query is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  query->query.group_topk_ = group_topk;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_group_query_set_include_vector(
    zvec_db_group_query_t *query, int include_vector) {
  if (!query) {
    return set_error("query is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  query->query.include_vector_ = include_vector != 0;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_group_query_set_filter(
    zvec_db_group_query_t *query, const char *filter) {
  if (!query || !filter) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  query->query.filter_ = filter;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_group_query_set_output_fields(
    zvec_db_group_query_t *query, const char **fields, size_t count) {
  if (!query) {
    return set_error("query is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  if (!fields) {
    query->query.output_fields_.reset();
    return ZVEC_STATUS_OK;
  }
  std::vector<std::string> out;
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    if (fields[i]) {
      out.emplace_back(fields[i]);
    }
  }
  query->query.output_fields_ = std::move(out);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_group_query_set_dense_vector_bytes(
    zvec_db_group_query_t *query, const void *data, size_t bytes) {
  if (!query || !data || bytes == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  query->query.query_vector_.assign(
      reinterpret_cast<const char *>(data), bytes);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_group_query_set_sparse_vector(
    zvec_db_group_query_t *query,
    const uint32_t *indices,
    const void *values,
    size_t count,
    size_t value_bytes) {
  if (!query || !indices || !values || count == 0 || value_bytes == 0) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  query->query.query_sparse_indices_.assign(
      reinterpret_cast<const char *>(indices),
      count * sizeof(uint32_t));
  query->query.query_sparse_values_.assign(
      reinterpret_cast<const char *>(values), value_bytes);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_group_query_set_params(
    zvec_db_group_query_t *query,
    const zvec_db_query_params_t *params) {
  if (!query || !params || !params->ptr) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  query->query.query_params_ = params->ptr;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_collection_groupby_query(
    zvec_db_collection_t *collection,
    const zvec_db_group_query_t *query,
    zvec_db_group_result_t **out_result) {
  if (!collection || !collection->ptr || !query || !out_result) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  auto result = collection->ptr->GroupByQuery(query->query);
  if (!result.has_value()) {
    return set_error(result.error().message());
  }
  auto *wrap = new zvec_db_group_result;
  wrap->groups = result.value();
  *out_result = wrap;
  return ZVEC_STATUS_OK;
}

extern "C" size_t zvec_db_group_result_group_count(
    const zvec_db_group_result_t *result) {
  if (!result) {
    set_error("result is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return result->groups.size();
}

extern "C" zvec_status_t zvec_db_group_result_get_group_value(
    const zvec_db_group_result_t *result,
    size_t group_idx,
    char **out_string) {
  if (!result || !out_string) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  if (group_idx >= result->groups.size()) {
    return set_error("index out of range", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  char *dup = dup_string(result->groups[group_idx].group_by_value_);
  if (!dup) {
    return set_error("failed to allocate string");
  }
  *out_string = dup;
  return ZVEC_STATUS_OK;
}

extern "C" size_t zvec_db_group_result_group_doc_count(
    const zvec_db_group_result_t *result,
    size_t group_idx) {
  if (!result) {
    set_error("result is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  if (group_idx >= result->groups.size()) {
    set_error("index out of range", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return result->groups[group_idx].docs_.size();
}

extern "C" zvec_status_t zvec_db_group_result_get_doc(
    const zvec_db_group_result_t *result,
    size_t group_idx,
    size_t doc_idx,
    zvec_db_doc_t **out_doc) {
  if (!result || !out_doc) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  if (group_idx >= result->groups.size()) {
    return set_error("index out of range", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  const auto &docs = result->groups[group_idx].docs_;
  if (doc_idx >= docs.size()) {
    return set_error("index out of range", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  auto doc = std::make_shared<zvec::Doc>(docs[doc_idx]);
  *out_doc = new zvec_db_doc{doc};
  return ZVEC_STATUS_OK;
}

extern "C" void zvec_db_group_result_destroy(
    zvec_db_group_result_t *result) {
  if (!result) {
    return;
  }
  delete result;
}

extern "C" zvec_status_t zvec_db_config_create(
    zvec_db_config_t **out_config) {
  if (!out_config) {
    return set_error("out_config is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  try {
    *out_config = new zvec_db_config();
    return ZVEC_STATUS_OK;
  } catch (const std::exception &e) {
    return set_error(e.what());
  }
}

extern "C" void zvec_db_config_destroy(zvec_db_config_t *config) {
  if (!config) {
    return;
  }
  delete config;
}

extern "C" zvec_status_t zvec_db_config_set_memory_limit_bytes(
    zvec_db_config_t *config,
    uint64_t value) {
  if (!config) {
    return set_error("config is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  config->data.memory_limit_bytes = value;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_config_set_query_thread_count(
    zvec_db_config_t *config,
    uint32_t value) {
  if (!config) {
    return set_error("config is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  config->data.query_thread_count = value;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_config_set_optimize_thread_count(
    zvec_db_config_t *config,
    uint32_t value) {
  if (!config) {
    return set_error("config is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  config->data.optimize_thread_count = value;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_config_set_invert_to_forward_scan_ratio(
    zvec_db_config_t *config,
    float value) {
  if (!config) {
    return set_error("config is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  config->data.invert_to_forward_scan_ratio = value;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_config_set_brute_force_by_keys_ratio(
    zvec_db_config_t *config,
    float value) {
  if (!config) {
    return set_error("config is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  config->data.brute_force_by_keys_ratio = value;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_config_set_console_logger(
    zvec_db_config_t *config,
    zvec_db_log_level_t level) {
  if (!config) {
    return set_error("config is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  auto log_level =
      static_cast<zvec::GlobalConfig::LogLevel>(level);
  config->data.log_config =
      std::make_shared<zvec::GlobalConfig::ConsoleLogConfig>(log_level);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_config_set_file_logger(
    zvec_db_config_t *config,
    zvec_db_log_level_t level,
    const char *dir,
    const char *basename,
    uint32_t file_size_mb,
    uint32_t overdue_days) {
  if (!config) {
    return set_error("config is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  auto log_level =
      static_cast<zvec::GlobalConfig::LogLevel>(level);
  std::string dir_str = dir ? dir : zvec::DEFAULT_LOG_DIR;
  std::string base_str = basename ? basename : zvec::DEFAULT_LOG_BASENAME;
  config->data.log_config =
      std::make_shared<zvec::GlobalConfig::FileLogConfig>(
          log_level, dir_str, base_str, file_size_mb, overdue_days);
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_db_global_config_init(
    const zvec_db_config_t *config) {
  if (!config) {
    return set_error("config is null", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  return status_from_db_status(
      zvec::GlobalConfig::Instance().Initialize(config->data));
}

extern "C" int zvec_ailego_string_starts_with(
    const char *ref,
    const char *prefix) {
  if (!ref || !prefix) {
    set_error("ref or prefix is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return zvec::ailego::StringHelper::StartsWith(ref, prefix) ? 1 : 0;
}

extern "C" int zvec_ailego_string_ends_with(
    const char *ref,
    const char *suffix) {
  if (!ref || !suffix) {
    set_error("ref or suffix is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return zvec::ailego::StringHelper::EndsWith(ref, suffix) ? 1 : 0;
}

extern "C" int zvec_ailego_string_compare_ignore_case(
    const char *a,
    const char *b) {
  if (!a || !b) {
    set_error("a or b is null", ZVEC_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  return zvec::ailego::StringHelper::CompareIgnoreCase(a, b) ? 1 : 0;
}

extern "C" zvec_status_t zvec_ailego_string_copy_trim(
    const char *input,
    char **out_string) {
  if (!input || !out_string) {
    return set_error("input or out_string is null",
                     ZVEC_STATUS_INVALID_ARGUMENT);
  }
  std::string out = zvec::ailego::StringHelper::CopyTrim(std::string(input));
  char *dup = dup_string(out);
  if (!dup) {
    return set_error("failed to allocate string");
  }
  *out_string = dup;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_ailego_string_copy_left_trim(
    const char *input,
    char **out_string) {
  if (!input || !out_string) {
    return set_error("input or out_string is null",
                     ZVEC_STATUS_INVALID_ARGUMENT);
  }
  std::string out =
      zvec::ailego::StringHelper::CopyLeftTrim(std::string(input));
  char *dup = dup_string(out);
  if (!dup) {
    return set_error("failed to allocate string");
  }
  *out_string = dup;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_ailego_string_copy_right_trim(
    const char *input,
    char **out_string) {
  if (!input || !out_string) {
    return set_error("input or out_string is null",
                     ZVEC_STATUS_INVALID_ARGUMENT);
  }
  std::string out =
      zvec::ailego::StringHelper::CopyRightTrim(std::string(input));
  char *dup = dup_string(out);
  if (!dup) {
    return set_error("failed to allocate string");
  }
  *out_string = dup;
  return ZVEC_STATUS_OK;
}

extern "C" zvec_status_t zvec_ailego_string_split(
    const char *input,
    const char *delim,
    char ***out_items,
    size_t *out_count) {
  if (!input || !delim || !out_items || !out_count) {
    return set_error("invalid arguments", ZVEC_STATUS_INVALID_ARGUMENT);
  }
  std::vector<std::string> parts;
  zvec::ailego::StringHelper::Split<std::string>(input, delim, &parts);
  if (parts.empty()) {
    *out_items = nullptr;
    *out_count = 0;
    return ZVEC_STATUS_OK;
  }
  char **items = static_cast<char **>(std::calloc(parts.size(), sizeof(char *)));
  if (!items) {
    return set_error("failed to allocate items");
  }
  for (size_t i = 0; i < parts.size(); ++i) {
    items[i] = dup_string(parts[i]);
    if (!items[i]) {
      for (size_t j = 0; j < i; ++j) {
        std::free(items[j]);
      }
      std::free(items);
      return set_error("failed to allocate item");
    }
  }
  *out_items = items;
  *out_count = parts.size();
  return ZVEC_STATUS_OK;
}

extern "C" void zvec_ailego_string_split_free(char **items, size_t count) {
  if (!items) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    std::free(items[i]);
  }
  std::free(items);
}

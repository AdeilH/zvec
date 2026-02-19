#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  ZVEC_STATUS_OK = 0,
  ZVEC_STATUS_ERR = 1,
  ZVEC_STATUS_INVALID_ARGUMENT = 2
} zvec_status_t;

// Global error handling (thread-local).
const char* zvec_last_error(void);
void zvec_clear_error(void);
void zvec_free(void* p);

// -----------------------
// Core interface (C API)
// -----------------------

typedef struct zvec_core_param zvec_core_param_t;
typedef struct zvec_core_query_param zvec_core_query_param_t;
typedef struct zvec_core_index zvec_core_index_t;
typedef struct zvec_core_search_result zvec_core_search_result_t;

typedef enum {
  ZVEC_CORE_INDEX_NONE = 0,
  ZVEC_CORE_INDEX_FLAT = 1,
  ZVEC_CORE_INDEX_IVF = 2,
  ZVEC_CORE_INDEX_HNSW = 3
} zvec_core_index_type_t;

typedef enum {
  ZVEC_CORE_METRIC_NONE = 0,
  ZVEC_CORE_METRIC_L2SQ = 1,
  ZVEC_CORE_METRIC_INNER_PRODUCT = 2,
  ZVEC_CORE_METRIC_COSINE = 3,
  ZVEC_CORE_METRIC_MIPS_L2SQ = 4
} zvec_core_metric_type_t;

typedef enum {
  ZVEC_CORE_DT_UNDEFINED = 0,
  ZVEC_CORE_DT_FP16 = 1,
  ZVEC_CORE_DT_FP32 = 2,
  ZVEC_CORE_DT_FP64 = 3,
  ZVEC_CORE_DT_INT8 = 4,
  ZVEC_CORE_DT_INT16 = 5,
  ZVEC_CORE_DT_INT4 = 6,
  ZVEC_CORE_DT_BINARY32 = 7,
  ZVEC_CORE_DT_BINARY64 = 8
} zvec_core_data_type_t;

typedef enum {
  ZVEC_CORE_STORAGE_NONE = 0,
  ZVEC_CORE_STORAGE_MMAP = 1,
  ZVEC_CORE_STORAGE_MEMORY = 2,
  ZVEC_CORE_STORAGE_BUFFER_POOL = 3
} zvec_core_storage_type_t;

#define ZVEC_CORE_DEFAULT_HNSW_EF_CONSTRUCTION 500
#define ZVEC_CORE_DEFAULT_HNSW_NEIGHBOR_CNT 50
#define ZVEC_CORE_DEFAULT_HNSW_EF_SEARCH 300

// Index parameter helpers
zvec_status_t zvec_core_param_create_hnsw(
    zvec_core_metric_type_t metric,
    zvec_core_data_type_t data_type,
    int dimension,
    int m,
    int ef_construction,
    int is_sparse,
    zvec_core_param_t** out_param);

zvec_status_t zvec_core_param_create_ivf(
    zvec_core_metric_type_t metric,
    zvec_core_data_type_t data_type,
    int dimension,
    int nlist,
    int niters,
    int use_soar,
    int is_sparse,
    zvec_core_param_t** out_param);

zvec_status_t zvec_core_param_create_flat(
    zvec_core_metric_type_t metric,
    zvec_core_data_type_t data_type,
    int dimension,
    int is_sparse,
    zvec_core_param_t** out_param);

zvec_status_t zvec_core_param_from_json(
    const char* json_str,
    zvec_core_param_t** out_param);

zvec_status_t zvec_core_param_to_json(
    const zvec_core_param_t* param,
    char** out_json);

void zvec_core_param_destroy(zvec_core_param_t* param);

// Query parameter helpers
zvec_status_t zvec_core_query_param_create_hnsw(
    int topk,
    int ef_search,
    int fetch_vector,
    float radius,
    int is_linear,
    zvec_core_query_param_t** out_param);

zvec_status_t zvec_core_query_param_create_ivf(
    int topk,
    int nprobe,
    int fetch_vector,
    float radius,
    int is_linear,
    zvec_core_query_param_t** out_param);

zvec_status_t zvec_core_query_param_create_flat(
    int topk,
    int fetch_vector,
    float radius,
    int is_linear,
    zvec_core_query_param_t** out_param);

zvec_status_t zvec_core_query_param_to_json(
    const zvec_core_query_param_t* param,
    char** out_json);

zvec_status_t zvec_core_query_param_from_json_hnsw(
    const char* json_str,
    zvec_core_query_param_t** out_param);

zvec_status_t zvec_core_query_param_from_json_ivf(
    const char* json_str,
    zvec_core_query_param_t** out_param);

zvec_status_t zvec_core_query_param_from_json_flat(
    const char* json_str,
    zvec_core_query_param_t** out_param);

void zvec_core_query_param_destroy(zvec_core_query_param_t* param);

// Index lifecycle
zvec_status_t zvec_core_index_create(
    const zvec_core_param_t* param,
    zvec_core_index_t** out_index);

zvec_status_t zvec_core_index_open(
    zvec_core_index_t* index,
    const char* path,
    zvec_core_storage_type_t storage_type,
    int create_new,
    int read_only);

zvec_status_t zvec_core_index_close(zvec_core_index_t* index);
zvec_status_t zvec_core_index_flush(zvec_core_index_t* index);
zvec_status_t zvec_core_index_train(zvec_core_index_t* index);
uint32_t zvec_core_index_doc_count(const zvec_core_index_t* index);
void zvec_core_index_destroy(zvec_core_index_t* index);

// Index operations (dense)
zvec_status_t zvec_core_index_add_dense(
    zvec_core_index_t* index,
    const void* data,
    size_t element_count,
    uint32_t doc_id);

zvec_status_t zvec_core_index_search_dense(
    zvec_core_index_t* index,
    const void* query,
    size_t element_count,
    const zvec_core_query_param_t* param,
    zvec_core_search_result_t** out_result);

// Index operations (sparse)
zvec_status_t zvec_core_index_add_sparse(
    zvec_core_index_t* index,
    const uint32_t* indices,
    const void* values,
    uint32_t count,
    uint32_t doc_id);

zvec_status_t zvec_core_index_search_sparse(
    zvec_core_index_t* index,
    const uint32_t* indices,
    const void* values,
    uint32_t count,
    const zvec_core_query_param_t* param,
    zvec_core_search_result_t** out_result);

zvec_status_t zvec_core_index_fetch_dense(
    zvec_core_index_t* index,
    uint32_t doc_id,
    void** out_data,
    size_t* out_bytes);

zvec_status_t zvec_core_index_fetch_sparse(
    zvec_core_index_t* index,
    uint32_t doc_id,
    uint32_t** out_indices,
    void** out_values,
    uint32_t* out_count,
    size_t* out_value_bytes);

int zvec_core_index_is_trained(const zvec_core_index_t* index);

zvec_status_t zvec_core_index_get_param_json(
    const zvec_core_index_t* index,
    char** out_json);

// Search result helpers
size_t zvec_core_search_result_size(const zvec_core_search_result_t* result);
zvec_status_t zvec_core_search_result_get(
    const zvec_core_search_result_t* result,
    size_t idx,
    uint64_t* out_key,
    float* out_score);
void zvec_core_search_result_destroy(zvec_core_search_result_t* result);

// Core framework factory helpers
typedef struct zvec_core_factory_object zvec_core_factory_object_t;

zvec_status_t zvec_core_factory_create_metric(
    const char* name,
    zvec_core_factory_object_t** out_obj);
int zvec_core_factory_has_metric(const char* name);
zvec_status_t zvec_core_factory_all_metrics(
    char*** out_names,
    size_t* out_count);

zvec_status_t zvec_core_factory_create_logger(
    const char* name,
    zvec_core_factory_object_t** out_obj);
int zvec_core_factory_has_logger(const char* name);
zvec_status_t zvec_core_factory_all_loggers(
    char*** out_names,
    size_t* out_count);

zvec_status_t zvec_core_factory_create_dumper(
    const char* name,
    zvec_core_factory_object_t** out_obj);
int zvec_core_factory_has_dumper(const char* name);
zvec_status_t zvec_core_factory_all_dumpers(
    char*** out_names,
    size_t* out_count);

zvec_status_t zvec_core_factory_create_container(
    const char* name,
    zvec_core_factory_object_t** out_obj);
int zvec_core_factory_has_container(const char* name);
zvec_status_t zvec_core_factory_all_containers(
    char*** out_names,
    size_t* out_count);

zvec_status_t zvec_core_factory_create_storage(
    const char* name,
    zvec_core_factory_object_t** out_obj);
int zvec_core_factory_has_storage(const char* name);
zvec_status_t zvec_core_factory_all_storages(
    char*** out_names,
    size_t* out_count);

zvec_status_t zvec_core_factory_create_converter(
    const char* name,
    zvec_core_factory_object_t** out_obj);
int zvec_core_factory_has_converter(const char* name);
zvec_status_t zvec_core_factory_all_converters(
    char*** out_names,
    size_t* out_count);

zvec_status_t zvec_core_factory_create_reformer(
    const char* name,
    zvec_core_factory_object_t** out_obj);
int zvec_core_factory_has_reformer(const char* name);
zvec_status_t zvec_core_factory_all_reformers(
    char*** out_names,
    size_t* out_count);

zvec_status_t zvec_core_factory_create_trainer(
    const char* name,
    zvec_core_factory_object_t** out_obj);
int zvec_core_factory_has_trainer(const char* name);
zvec_status_t zvec_core_factory_all_trainers(
    char*** out_names,
    size_t* out_count);

zvec_status_t zvec_core_factory_create_builder(
    const char* name,
    zvec_core_factory_object_t** out_obj);
int zvec_core_factory_has_builder(const char* name);
zvec_status_t zvec_core_factory_all_builders(
    char*** out_names,
    size_t* out_count);

zvec_status_t zvec_core_factory_create_searcher(
    const char* name,
    zvec_core_factory_object_t** out_obj);
int zvec_core_factory_has_searcher(const char* name);
zvec_status_t zvec_core_factory_all_searchers(
    char*** out_names,
    size_t* out_count);

zvec_status_t zvec_core_factory_create_streamer(
    const char* name,
    zvec_core_factory_object_t** out_obj);
int zvec_core_factory_has_streamer(const char* name);
zvec_status_t zvec_core_factory_all_streamers(
    char*** out_names,
    size_t* out_count);

zvec_status_t zvec_core_factory_create_reducer(
    const char* name,
    zvec_core_factory_object_t** out_obj);
int zvec_core_factory_has_reducer(const char* name);
zvec_status_t zvec_core_factory_all_reducers(
    char*** out_names,
    size_t* out_count);

zvec_status_t zvec_core_factory_create_cluster(
    const char* name,
    zvec_core_factory_object_t** out_obj);
int zvec_core_factory_has_cluster(const char* name);
zvec_status_t zvec_core_factory_all_clusters(
    char*** out_names,
    size_t* out_count);

zvec_status_t zvec_core_factory_create_streamer_reducer(
    const char* name,
    zvec_core_factory_object_t** out_obj);
int zvec_core_factory_has_streamer_reducer(const char* name);
zvec_status_t zvec_core_factory_all_streamer_reducers(
    char*** out_names,
    size_t* out_count);

zvec_status_t zvec_core_factory_create_refiner(
    const char* name,
    zvec_core_factory_object_t** out_obj);
int zvec_core_factory_has_refiner(const char* name);
zvec_status_t zvec_core_factory_all_refiners(
    char*** out_names,
    size_t* out_count);

void zvec_core_factory_object_destroy(zvec_core_factory_object_t* obj);
void zvec_core_factory_string_array_free(char** items, size_t count);

// -----------------------
// DB interface (C API)
// -----------------------

typedef struct zvec_db_schema zvec_db_schema_t;
typedef struct zvec_db_field zvec_db_field_t;
typedef struct zvec_db_index_params zvec_db_index_params_t;
typedef struct zvec_db_doc zvec_db_doc_t;
typedef struct zvec_db_collection zvec_db_collection_t;
typedef struct zvec_db_query_params zvec_db_query_params_t;
typedef struct zvec_db_query zvec_db_query_t;
typedef struct zvec_db_query_result zvec_db_query_result_t;

typedef enum {
  ZVEC_DB_DT_UNDEFINED = 0,
  ZVEC_DB_DT_BINARY = 1,
  ZVEC_DB_DT_STRING = 2,
  ZVEC_DB_DT_BOOL = 3,
  ZVEC_DB_DT_INT32 = 4,
  ZVEC_DB_DT_INT64 = 5,
  ZVEC_DB_DT_UINT32 = 6,
  ZVEC_DB_DT_UINT64 = 7,
  ZVEC_DB_DT_FLOAT = 8,
  ZVEC_DB_DT_DOUBLE = 9,
  ZVEC_DB_DT_VECTOR_BINARY32 = 20,
  ZVEC_DB_DT_VECTOR_BINARY64 = 21,
  ZVEC_DB_DT_VECTOR_FP16 = 22,
  ZVEC_DB_DT_VECTOR_FP32 = 23,
  ZVEC_DB_DT_VECTOR_FP64 = 24,
  ZVEC_DB_DT_VECTOR_INT4 = 25,
  ZVEC_DB_DT_VECTOR_INT8 = 26,
  ZVEC_DB_DT_VECTOR_INT16 = 27,
  ZVEC_DB_DT_SPARSE_VECTOR_FP16 = 30,
  ZVEC_DB_DT_SPARSE_VECTOR_FP32 = 31,
  ZVEC_DB_DT_ARRAY_BINARY = 40,
  ZVEC_DB_DT_ARRAY_STRING = 41,
  ZVEC_DB_DT_ARRAY_BOOL = 42,
  ZVEC_DB_DT_ARRAY_INT32 = 43,
  ZVEC_DB_DT_ARRAY_INT64 = 44,
  ZVEC_DB_DT_ARRAY_UINT32 = 45,
  ZVEC_DB_DT_ARRAY_UINT64 = 46,
  ZVEC_DB_DT_ARRAY_FLOAT = 47,
  ZVEC_DB_DT_ARRAY_DOUBLE = 48
} zvec_db_data_type_t;

typedef enum {
  ZVEC_DB_STATUS_OK = 0,
  ZVEC_DB_STATUS_NOT_FOUND = 1,
  ZVEC_DB_STATUS_ALREADY_EXISTS = 2,
  ZVEC_DB_STATUS_INVALID_ARGUMENT = 3,
  ZVEC_DB_STATUS_PERMISSION_DENIED = 4,
  ZVEC_DB_STATUS_FAILED_PRECONDITION = 5,
  ZVEC_DB_STATUS_RESOURCE_EXHAUSTED = 6,
  ZVEC_DB_STATUS_UNAVAILABLE = 7,
  ZVEC_DB_STATUS_INTERNAL_ERROR = 8,
  ZVEC_DB_STATUS_NOT_SUPPORTED = 9,
  ZVEC_DB_STATUS_UNKNOWN = 10
} zvec_db_status_code_t;

const char* zvec_db_status_default_message(zvec_db_status_code_t code);

typedef enum {
  ZVEC_DB_INDEX_UNDEFINED = 0,
  ZVEC_DB_INDEX_HNSW = 1,
  ZVEC_DB_INDEX_IVF = 3,
  ZVEC_DB_INDEX_FLAT = 4,
  ZVEC_DB_INDEX_INVERT = 10
} zvec_db_index_type_t;

typedef enum {
  ZVEC_DB_QT_UNDEFINED = 0,
  ZVEC_DB_QT_FP16 = 1,
  ZVEC_DB_QT_INT8 = 2,
  ZVEC_DB_QT_INT4 = 3
} zvec_db_quantize_type_t;

#define ZVEC_DB_QUANTIZE_NONE ZVEC_DB_QT_UNDEFINED
#define ZVEC_DB_QUANTIZE_FP16 ZVEC_DB_QT_FP16
#define ZVEC_DB_QUANTIZE_INT8 ZVEC_DB_QT_INT8
#define ZVEC_DB_QUANTIZE_INT4 ZVEC_DB_QT_INT4

typedef enum {
  ZVEC_DB_METRIC_UNDEFINED = 0,
  ZVEC_DB_METRIC_L2 = 1,
  ZVEC_DB_METRIC_IP = 2,
  ZVEC_DB_METRIC_COSINE = 3,
  ZVEC_DB_METRIC_MIPSL2 = 4
} zvec_db_metric_type_t;

typedef enum {
  ZVEC_DB_OP_INSERT = 0,
  ZVEC_DB_OP_UPSERT = 1,
  ZVEC_DB_OP_UPDATE = 2,
  ZVEC_DB_OP_DELETE = 3
} zvec_db_operator_t;

typedef enum {
  ZVEC_DB_COMPARE_NONE = 0,
  ZVEC_DB_COMPARE_EQ = 1,
  ZVEC_DB_COMPARE_NE = 2,
  ZVEC_DB_COMPARE_LT = 3,
  ZVEC_DB_COMPARE_LE = 4,
  ZVEC_DB_COMPARE_GT = 5,
  ZVEC_DB_COMPARE_GE = 6,
  ZVEC_DB_COMPARE_LIKE = 7,
  ZVEC_DB_COMPARE_CONTAIN_ALL = 8,
  ZVEC_DB_COMPARE_CONTAIN_ANY = 9,
  ZVEC_DB_COMPARE_NOT_CONTAIN_ALL = 10,
  ZVEC_DB_COMPARE_NOT_CONTAIN_ANY = 11,
  ZVEC_DB_COMPARE_IS_NULL = 12,
  ZVEC_DB_COMPARE_IS_NOT_NULL = 13,
  ZVEC_DB_COMPARE_HAS_PREFIX = 14,
  ZVEC_DB_COMPARE_HAS_SUFFIX = 15
} zvec_db_compare_op_t;

typedef enum {
  ZVEC_DB_RELATION_NONE = 0,
  ZVEC_DB_RELATION_AND = 1,
  ZVEC_DB_RELATION_OR = 2
} zvec_db_relation_op_t;

typedef enum {
  ZVEC_DB_BLOCK_UNDEFINED = 0,
  ZVEC_DB_BLOCK_SCALAR = 1,
  ZVEC_DB_BLOCK_SCALAR_INDEX = 2,
  ZVEC_DB_BLOCK_VECTOR_INDEX = 3,
  ZVEC_DB_BLOCK_VECTOR_INDEX_QUANTIZE = 4
} zvec_db_block_type_t;

typedef enum {
  ZVEC_DB_FILE_UNKNOWN = 0,
  ZVEC_DB_FILE_IPC = 1,
  ZVEC_DB_FILE_PARQUET = 2
} zvec_db_file_format_t;

typedef enum {
  ZVEC_DB_COLUMN_UNDEFINED = 0,
  ZVEC_DB_COLUMN_ADD = 1,
  ZVEC_DB_COLUMN_ALTER = 2,
  ZVEC_DB_COLUMN_DROP = 3
} zvec_db_column_op_t;

#define ZVEC_DB_DEFAULT_MAX_BUFFER_SIZE (64U * 1024U * 1024U)
#define ZVEC_DB_MAX_DOC_COUNT_PER_SEGMENT 10000000ULL
#define ZVEC_DB_MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD 1000ULL

#define ZVEC_DB_MIN_LOG_FILE_SIZE 128U
#define ZVEC_DB_DEFAULT_LOG_FILE_SIZE 2048U
#define ZVEC_DB_DEFAULT_LOG_OVERDUE_DAYS 7U

// Schema / Field / Index params
zvec_status_t zvec_db_schema_create(const char* name, zvec_db_schema_t** out);
zvec_status_t zvec_db_schema_set_max_docs_per_segment(
    zvec_db_schema_t* schema,
    uint64_t max_docs);
zvec_status_t zvec_db_schema_add_field(
    zvec_db_schema_t* schema,
    const zvec_db_field_t* field);
int zvec_db_schema_has_field(
    const zvec_db_schema_t* schema,
    const char* name);
zvec_status_t zvec_db_schema_drop_field(
    zvec_db_schema_t* schema,
    const char* name);
zvec_status_t zvec_db_schema_alter_field(
    zvec_db_schema_t* schema,
    const char* name,
    const zvec_db_field_t* new_field);
zvec_status_t zvec_db_schema_add_index(
    zvec_db_schema_t* schema,
    const char* field,
    const zvec_db_index_params_t* params);
zvec_status_t zvec_db_schema_drop_index(
    zvec_db_schema_t* schema,
    const char* field);
int zvec_db_schema_has_index(
    const zvec_db_schema_t* schema,
    const char* field);
zvec_status_t zvec_db_schema_get_name(
    const zvec_db_schema_t* schema,
    char** out_name);
uint64_t zvec_db_schema_get_max_docs_per_segment(
    const zvec_db_schema_t* schema);
zvec_status_t zvec_db_schema_to_string(
    const zvec_db_schema_t* schema,
    char** out_string);
zvec_status_t zvec_db_schema_to_formatted_string(
    const zvec_db_schema_t* schema,
    int indent_level,
    char** out_string);
zvec_status_t zvec_db_schema_validate(const zvec_db_schema_t* schema);
size_t zvec_db_schema_field_count(const zvec_db_schema_t* schema);
zvec_status_t zvec_db_schema_get_field(
    const zvec_db_schema_t* schema,
    size_t index,
    zvec_db_field_t** out_field);
zvec_status_t zvec_db_schema_get_field_by_name(
    const zvec_db_schema_t* schema,
    const char* name,
    zvec_db_field_t** out_field);
zvec_status_t zvec_db_schema_field_names(
    const zvec_db_schema_t* schema,
    char*** out_names,
    size_t* out_count);
// void zvec_db_schema_destroy(zvec_db_schema_t* schema);

  void zvec_db_schema_destroy(zvec_db_schema_t* schema);
  size_t zvec_db_schema_sizeof(void);

zvec_status_t zvec_db_field_create_scalar(
    const char* name,
    zvec_db_data_type_t data_type,
    int nullable,
    const zvec_db_index_params_t* index_params,
    zvec_db_field_t** out_field);

zvec_status_t zvec_db_field_create_vector(
    const char* name,
    zvec_db_data_type_t data_type,
    uint32_t dimension,
    int nullable,
    const zvec_db_index_params_t* index_params,
    zvec_db_field_t** out_field);

zvec_status_t zvec_db_field_create_sparse_vector(
    const char* name,
    zvec_db_data_type_t data_type,
    uint32_t dimension,
    int nullable,
    const zvec_db_index_params_t* index_params,
    zvec_db_field_t** out_field);

zvec_status_t zvec_db_field_get_name(
    const zvec_db_field_t* field,
    char** out_name);
zvec_status_t zvec_db_field_get_data_type(
    const zvec_db_field_t* field,
    zvec_db_data_type_t* out_type);
zvec_status_t zvec_db_field_get_dimension(
    const zvec_db_field_t* field,
    uint32_t* out_dimension);
zvec_status_t zvec_db_field_get_nullable(
    const zvec_db_field_t* field,
    int* out_nullable);
zvec_status_t zvec_db_field_get_index_type(
    const zvec_db_field_t* field,
    zvec_db_index_type_t* out_type);
zvec_status_t zvec_db_field_get_index_params(
    const zvec_db_field_t* field,
    zvec_db_index_params_t** out_params);
int zvec_db_field_is_vector_field(const zvec_db_field_t* field);
int zvec_db_field_is_dense_vector(const zvec_db_field_t* field);
int zvec_db_field_is_sparse_vector(const zvec_db_field_t* field);
int zvec_db_field_is_array_type(const zvec_db_field_t* field);
zvec_status_t zvec_db_field_to_string(
    const zvec_db_field_t* field,
    char** out_string);
zvec_status_t zvec_db_field_to_formatted_string(
    const zvec_db_field_t* field,
    int indent_level,
    char** out_string);
zvec_status_t zvec_db_field_validate(const zvec_db_field_t* field);
void zvec_db_field_destroy(zvec_db_field_t* field);

zvec_status_t zvec_db_index_params_create_invert(
    int enable_range_optimization,
    int enable_extended_wildcard,
    zvec_db_index_params_t** out_params);

zvec_status_t zvec_db_index_params_create_hnsw(
    zvec_db_metric_type_t metric,
    int m,
    int ef_construction,
    zvec_db_quantize_type_t quantize,
    zvec_db_index_params_t** out_params);

zvec_status_t zvec_db_index_params_create_ivf(
    zvec_db_metric_type_t metric,
    int n_list,
    int n_iters,
    int use_soar,
    zvec_db_quantize_type_t quantize,
    zvec_db_index_params_t** out_params);

zvec_status_t zvec_db_index_params_create_flat(
    zvec_db_metric_type_t metric,
    zvec_db_quantize_type_t quantize,
    zvec_db_index_params_t** out_params);

zvec_status_t zvec_db_index_params_type(
    const zvec_db_index_params_t* params,
    zvec_db_index_type_t* out_type);
zvec_status_t zvec_db_index_params_to_string(
    const zvec_db_index_params_t* params,
    char** out_string);
zvec_status_t zvec_db_index_params_get_metric(
    const zvec_db_index_params_t* params,
    zvec_db_metric_type_t* out_metric);
zvec_status_t zvec_db_index_params_get_quantize(
    const zvec_db_index_params_t* params,
    zvec_db_quantize_type_t* out_quantize);
zvec_status_t zvec_db_index_params_get_hnsw_m(
    const zvec_db_index_params_t* params,
    int* out_m);
zvec_status_t zvec_db_index_params_get_hnsw_ef_construction(
    const zvec_db_index_params_t* params,
    int* out_ef_construction);
zvec_status_t zvec_db_index_params_get_ivf_nlist(
    const zvec_db_index_params_t* params,
    int* out_n_list);
zvec_status_t zvec_db_index_params_get_ivf_niters(
    const zvec_db_index_params_t* params,
    int* out_n_iters);
zvec_status_t zvec_db_index_params_get_ivf_use_soar(
    const zvec_db_index_params_t* params,
    int* out_use_soar);
zvec_status_t zvec_db_index_params_get_invert_range_optimization(
    const zvec_db_index_params_t* params,
    int* out_enabled);
zvec_status_t zvec_db_index_params_get_invert_extended_wildcard(
    const zvec_db_index_params_t* params,
    int* out_enabled);
void zvec_db_index_params_destroy(zvec_db_index_params_t* params);

// Docs
zvec_status_t zvec_db_doc_create(zvec_db_doc_t** out_doc);
void zvec_db_doc_destroy(zvec_db_doc_t* doc);
zvec_status_t zvec_db_doc_set_pk(zvec_db_doc_t* doc, const char* pk);
zvec_status_t zvec_db_doc_set_doc_id(zvec_db_doc_t* doc, uint64_t doc_id);
zvec_status_t zvec_db_doc_set_score(zvec_db_doc_t* doc, float score);
zvec_status_t zvec_db_doc_set_operator(zvec_db_doc_t* doc, zvec_db_operator_t op);

zvec_status_t zvec_db_doc_set_bool(
    zvec_db_doc_t* doc, const char* field, int value);
zvec_status_t zvec_db_doc_set_int32(
    zvec_db_doc_t* doc, const char* field, int32_t value);
zvec_status_t zvec_db_doc_set_int64(
    zvec_db_doc_t* doc, const char* field, int64_t value);
zvec_status_t zvec_db_doc_set_uint32(
    zvec_db_doc_t* doc, const char* field, uint32_t value);
zvec_status_t zvec_db_doc_set_uint64(
    zvec_db_doc_t* doc, const char* field, uint64_t value);
zvec_status_t zvec_db_doc_set_float(
    zvec_db_doc_t* doc, const char* field, float value);
zvec_status_t zvec_db_doc_set_double(
    zvec_db_doc_t* doc, const char* field, double value);
zvec_status_t zvec_db_doc_set_string(
    zvec_db_doc_t* doc, const char* field, const char* value);
zvec_status_t zvec_db_doc_set_binary(
    zvec_db_doc_t* doc, const char* field, const void* data, size_t size);
zvec_status_t zvec_db_doc_set_null(
    zvec_db_doc_t* doc, const char* field);
zvec_status_t zvec_db_doc_remove_field(
    zvec_db_doc_t* doc, const char* field);
void zvec_db_doc_clear(zvec_db_doc_t* doc);

zvec_status_t zvec_db_doc_set_vector_fp32(
    zvec_db_doc_t* doc, const char* field,
    const float* values, size_t count);
zvec_status_t zvec_db_doc_set_vector_fp64(
    zvec_db_doc_t* doc, const char* field,
    const double* values, size_t count);
zvec_status_t zvec_db_doc_set_vector_fp16(
    zvec_db_doc_t* doc, const char* field,
    const float* values, size_t count);
zvec_status_t zvec_db_doc_set_vector_int8(
    zvec_db_doc_t* doc, const char* field,
    const int8_t* values, size_t count);
zvec_status_t zvec_db_doc_set_vector_int16(
    zvec_db_doc_t* doc, const char* field,
    const int16_t* values, size_t count);
zvec_status_t zvec_db_doc_set_vector_int32(
    zvec_db_doc_t* doc, const char* field,
    const int32_t* values, size_t count);
zvec_status_t zvec_db_doc_set_vector_int64(
    zvec_db_doc_t* doc, const char* field,
    const int64_t* values, size_t count);
zvec_status_t zvec_db_doc_set_vector_uint32(
    zvec_db_doc_t* doc, const char* field,
    const uint32_t* values, size_t count);
zvec_status_t zvec_db_doc_set_vector_uint64(
    zvec_db_doc_t* doc, const char* field,
    const uint64_t* values, size_t count);
zvec_status_t zvec_db_doc_set_vector_binary32(
    zvec_db_doc_t* doc, const char* field,
    const uint32_t* values, size_t count);
zvec_status_t zvec_db_doc_set_vector_binary64(
    zvec_db_doc_t* doc, const char* field,
    const uint64_t* values, size_t count);
zvec_status_t zvec_db_doc_set_vector_int4(
    zvec_db_doc_t* doc, const char* field,
    const int8_t* values, size_t count);

zvec_status_t zvec_db_doc_set_sparse_vector_fp32(
    zvec_db_doc_t* doc, const char* field,
    const uint32_t* indices,
    const float* values,
    size_t count);
zvec_status_t zvec_db_doc_set_sparse_vector_fp16(
    zvec_db_doc_t* doc, const char* field,
    const uint32_t* indices,
    const float* values,
    size_t count);

zvec_status_t zvec_db_doc_set_array_bool(
    zvec_db_doc_t* doc, const char* field,
    const uint8_t* values, size_t count);
zvec_status_t zvec_db_doc_set_array_int32(
    zvec_db_doc_t* doc, const char* field,
    const int32_t* values, size_t count);
zvec_status_t zvec_db_doc_set_array_int64(
    zvec_db_doc_t* doc, const char* field,
    const int64_t* values, size_t count);
zvec_status_t zvec_db_doc_set_array_uint32(
    zvec_db_doc_t* doc, const char* field,
    const uint32_t* values, size_t count);
zvec_status_t zvec_db_doc_set_array_uint64(
    zvec_db_doc_t* doc, const char* field,
    const uint64_t* values, size_t count);
zvec_status_t zvec_db_doc_set_array_float(
    zvec_db_doc_t* doc, const char* field,
    const float* values, size_t count);
zvec_status_t zvec_db_doc_set_array_double(
    zvec_db_doc_t* doc, const char* field,
    const double* values, size_t count);
zvec_status_t zvec_db_doc_set_array_string(
    zvec_db_doc_t* doc, const char* field,
    const char** values, size_t count);
zvec_status_t zvec_db_doc_set_array_binary(
    zvec_db_doc_t* doc, const char* field,
    const char** values, size_t count);

zvec_status_t zvec_db_doc_get_pk(
    const zvec_db_doc_t* doc, char** out_pk);
zvec_status_t zvec_db_doc_get_doc_id(
    const zvec_db_doc_t* doc, uint64_t* out_doc_id);
zvec_status_t zvec_db_doc_get_score(
    const zvec_db_doc_t* doc, float* out_score);
zvec_status_t zvec_db_doc_get_operator(
    const zvec_db_doc_t* doc, zvec_db_operator_t* out_op);
int zvec_db_doc_has(const zvec_db_doc_t* doc, const char* field);
int zvec_db_doc_has_value(const zvec_db_doc_t* doc, const char* field);
int zvec_db_doc_is_null(const zvec_db_doc_t* doc, const char* field);
int zvec_db_doc_is_empty(const zvec_db_doc_t* doc);
zvec_status_t zvec_db_doc_field_names(
    const zvec_db_doc_t* doc,
    char*** out_names,
    size_t* out_count);

zvec_status_t zvec_db_doc_get_bool(
    const zvec_db_doc_t* doc, const char* field, int* out_value);
zvec_status_t zvec_db_doc_get_int32(
    const zvec_db_doc_t* doc, const char* field, int32_t* out_value);
zvec_status_t zvec_db_doc_get_int64(
    const zvec_db_doc_t* doc, const char* field, int64_t* out_value);
zvec_status_t zvec_db_doc_get_uint32(
    const zvec_db_doc_t* doc, const char* field, uint32_t* out_value);
zvec_status_t zvec_db_doc_get_uint64(
    const zvec_db_doc_t* doc, const char* field, uint64_t* out_value);
zvec_status_t zvec_db_doc_get_float(
    const zvec_db_doc_t* doc, const char* field, float* out_value);
zvec_status_t zvec_db_doc_get_double(
    const zvec_db_doc_t* doc, const char* field, double* out_value);
zvec_status_t zvec_db_doc_get_string(
    const zvec_db_doc_t* doc, const char* field, char** out_string);
zvec_status_t zvec_db_doc_get_binary(
    const zvec_db_doc_t* doc, const char* field,
    void** out_data, size_t* out_size);

zvec_status_t zvec_db_doc_get_vector_fp32(
    const zvec_db_doc_t* doc, const char* field,
    float** out_values, size_t* out_count);
zvec_status_t zvec_db_doc_get_vector_fp64(
    const zvec_db_doc_t* doc, const char* field,
    double** out_values, size_t* out_count);
zvec_status_t zvec_db_doc_get_vector_fp16(
    const zvec_db_doc_t* doc, const char* field,
    float** out_values, size_t* out_count);
zvec_status_t zvec_db_doc_get_vector_int8(
    const zvec_db_doc_t* doc, const char* field,
    int8_t** out_values, size_t* out_count);
zvec_status_t zvec_db_doc_get_vector_int16(
    const zvec_db_doc_t* doc, const char* field,
    int16_t** out_values, size_t* out_count);
zvec_status_t zvec_db_doc_get_vector_int32(
    const zvec_db_doc_t* doc, const char* field,
    int32_t** out_values, size_t* out_count);
zvec_status_t zvec_db_doc_get_vector_int64(
    const zvec_db_doc_t* doc, const char* field,
    int64_t** out_values, size_t* out_count);
zvec_status_t zvec_db_doc_get_vector_uint32(
    const zvec_db_doc_t* doc, const char* field,
    uint32_t** out_values, size_t* out_count);
zvec_status_t zvec_db_doc_get_vector_uint64(
    const zvec_db_doc_t* doc, const char* field,
    uint64_t** out_values, size_t* out_count);
zvec_status_t zvec_db_doc_get_vector_binary32(
    const zvec_db_doc_t* doc, const char* field,
    uint32_t** out_values, size_t* out_count);
zvec_status_t zvec_db_doc_get_vector_binary64(
    const zvec_db_doc_t* doc, const char* field,
    uint64_t** out_values, size_t* out_count);
zvec_status_t zvec_db_doc_get_vector_int4(
    const zvec_db_doc_t* doc, const char* field,
    int8_t** out_values, size_t* out_count);

zvec_status_t zvec_db_doc_get_sparse_vector_fp32(
    const zvec_db_doc_t* doc, const char* field,
    uint32_t** out_indices,
    float** out_values,
    size_t* out_count);
zvec_status_t zvec_db_doc_get_sparse_vector_fp16(
    const zvec_db_doc_t* doc, const char* field,
    uint32_t** out_indices,
    float** out_values,
    size_t* out_count);

zvec_status_t zvec_db_doc_get_array_bool(
    const zvec_db_doc_t* doc, const char* field,
    uint8_t** out_values, size_t* out_count);
zvec_status_t zvec_db_doc_get_array_int32(
    const zvec_db_doc_t* doc, const char* field,
    int32_t** out_values, size_t* out_count);
zvec_status_t zvec_db_doc_get_array_int64(
    const zvec_db_doc_t* doc, const char* field,
    int64_t** out_values, size_t* out_count);
zvec_status_t zvec_db_doc_get_array_uint32(
    const zvec_db_doc_t* doc, const char* field,
    uint32_t** out_values, size_t* out_count);
zvec_status_t zvec_db_doc_get_array_uint64(
    const zvec_db_doc_t* doc, const char* field,
    uint64_t** out_values, size_t* out_count);
zvec_status_t zvec_db_doc_get_array_float(
    const zvec_db_doc_t* doc, const char* field,
    float** out_values, size_t* out_count);
zvec_status_t zvec_db_doc_get_array_double(
    const zvec_db_doc_t* doc, const char* field,
    double** out_values, size_t* out_count);
zvec_status_t zvec_db_doc_get_array_string(
    const zvec_db_doc_t* doc, const char* field,
    char*** out_values, size_t* out_count);
zvec_status_t zvec_db_doc_get_array_binary(
    const zvec_db_doc_t* doc, const char* field,
    void*** out_values, size_t** out_sizes, size_t* out_count);

size_t zvec_db_doc_memory_usage(const zvec_db_doc_t* doc);
zvec_status_t zvec_db_doc_validate(
    const zvec_db_doc_t* doc,
    const zvec_db_schema_t* schema,
    int is_update);

void zvec_db_string_array_free(char** items, size_t count);
void zvec_db_binary_array_free(void** items, size_t* sizes, size_t count);

zvec_status_t zvec_db_doc_to_string(
    const zvec_db_doc_t* doc,
    char** out_string);
zvec_status_t zvec_db_doc_to_detail_string(
    const zvec_db_doc_t* doc,
    char** out_string);

// Collection
zvec_status_t zvec_db_collection_create_and_open(
    const char* path,
    const zvec_db_schema_t* schema,
    int read_only,
    int enable_mmap,
    uint32_t max_buffer_size,
    zvec_db_collection_t** out_collection);

zvec_status_t zvec_db_collection_open(
    const char* path,
    int read_only,
    int enable_mmap,
    uint32_t max_buffer_size,
    zvec_db_collection_t** out_collection);

void zvec_db_collection_destroy(zvec_db_collection_t* collection);
zvec_status_t zvec_db_collection_destroy_storage(
    zvec_db_collection_t* collection);

zvec_status_t zvec_db_collection_flush(zvec_db_collection_t* collection);
zvec_status_t zvec_db_collection_optimize(zvec_db_collection_t* collection);
zvec_status_t zvec_db_collection_stats(
    zvec_db_collection_t* collection,
    char** out_string);
zvec_status_t zvec_db_collection_path(
    zvec_db_collection_t* collection,
    char** out_string);
zvec_status_t zvec_db_collection_get_schema(
    zvec_db_collection_t* collection,
    zvec_db_schema_t** out_schema);
zvec_status_t zvec_db_collection_get_options(
    zvec_db_collection_t* collection,
    int* out_read_only,
    int* out_enable_mmap,
    uint32_t* out_max_buffer_size);

zvec_status_t zvec_db_collection_create_index(
    zvec_db_collection_t* collection,
    const char* field,
    const zvec_db_index_params_t* params,
    int concurrency);
zvec_status_t zvec_db_collection_drop_index(
    zvec_db_collection_t* collection,
    const char* field);
zvec_status_t zvec_db_collection_add_column(
    zvec_db_collection_t* collection,
    const zvec_db_field_t* field,
    const char* expression,
    int concurrency);
zvec_status_t zvec_db_collection_drop_column(
    zvec_db_collection_t* collection,
    const char* field);
zvec_status_t zvec_db_collection_alter_column(
    zvec_db_collection_t* collection,
    const char* field,
    const char* rename,
    const zvec_db_field_t* new_field,
    int concurrency);

zvec_status_t zvec_db_collection_insert(
    zvec_db_collection_t* collection,
    zvec_db_doc_t** docs,
    size_t count);

zvec_status_t zvec_db_collection_upsert(
    zvec_db_collection_t* collection,
    zvec_db_doc_t** docs,
    size_t count);

zvec_status_t zvec_db_collection_update(
    zvec_db_collection_t* collection,
    zvec_db_doc_t** docs,
    size_t count);
zvec_status_t zvec_db_collection_delete(
    zvec_db_collection_t* collection,
    const char** pks,
    size_t count);
zvec_status_t zvec_db_collection_delete_by_filter(
    zvec_db_collection_t* collection,
    const char* filter);

// Query params
zvec_status_t zvec_db_query_params_create_hnsw(
    int ef,
    float radius,
    int is_linear,
    int is_using_refiner,
    zvec_db_query_params_t** out_params);

zvec_status_t zvec_db_query_params_create_ivf(
    int nprobe,
    float scale_factor,
    int is_using_refiner,
    zvec_db_query_params_t** out_params);

zvec_status_t zvec_db_query_params_create_flat(
    float scale_factor,
    int is_using_refiner,
    zvec_db_query_params_t** out_params);

zvec_status_t zvec_db_query_params_set_radius(
    zvec_db_query_params_t* params, float radius);
zvec_status_t zvec_db_query_params_get_radius(
    const zvec_db_query_params_t* params, float* out_radius);
zvec_status_t zvec_db_query_params_set_is_linear(
    zvec_db_query_params_t* params, int is_linear);
zvec_status_t zvec_db_query_params_get_is_linear(
    const zvec_db_query_params_t* params, int* out_is_linear);
zvec_status_t zvec_db_query_params_set_is_using_refiner(
    zvec_db_query_params_t* params, int is_using_refiner);
zvec_status_t zvec_db_query_params_get_is_using_refiner(
    const zvec_db_query_params_t* params, int* out_is_using_refiner);
zvec_status_t zvec_db_query_params_set_hnsw_ef(
    zvec_db_query_params_t* params, int ef);
zvec_status_t zvec_db_query_params_get_hnsw_ef(
    const zvec_db_query_params_t* params, int* out_ef);
zvec_status_t zvec_db_query_params_set_ivf_nprobe(
    zvec_db_query_params_t* params, int nprobe);
zvec_status_t zvec_db_query_params_get_ivf_nprobe(
    const zvec_db_query_params_t* params, int* out_nprobe);
zvec_status_t zvec_db_query_params_set_scale_factor(
    zvec_db_query_params_t* params, float scale_factor);
zvec_status_t zvec_db_query_params_get_scale_factor(
    const zvec_db_query_params_t* params, float* out_scale_factor);

void zvec_db_query_params_destroy(zvec_db_query_params_t* params);

// Query
zvec_status_t zvec_db_query_create(
    const char* field_name,
    zvec_db_query_t** out_query);
void zvec_db_query_destroy(zvec_db_query_t* query);
zvec_status_t zvec_db_query_set_topk(zvec_db_query_t* query, int topk);
zvec_status_t zvec_db_query_set_include_vector(
    zvec_db_query_t* query, int include_vector);
zvec_status_t zvec_db_query_set_include_doc_id(
    zvec_db_query_t* query, int include_doc_id);
zvec_status_t zvec_db_query_set_filter(
    zvec_db_query_t* query, const char* filter);
zvec_status_t zvec_db_query_set_output_fields(
    zvec_db_query_t* query, const char** fields, size_t count);
zvec_status_t zvec_db_query_set_dense_vector_bytes(
    zvec_db_query_t* query, const void* data, size_t bytes);
zvec_status_t zvec_db_query_set_sparse_vector(
    zvec_db_query_t* query,
    const uint32_t* indices,
    const void* values,
    size_t count,
    size_t value_bytes);
zvec_status_t zvec_db_query_set_params(
    zvec_db_query_t* query,
    const zvec_db_query_params_t* params);

// Query execution
zvec_status_t zvec_db_collection_query(
    zvec_db_collection_t* collection,
    const zvec_db_query_t* query,
    zvec_db_query_result_t** out_result);

zvec_status_t zvec_db_collection_fetch(
    zvec_db_collection_t* collection,
    const char** pks,
    size_t count,
    zvec_db_query_result_t** out_result);

size_t zvec_db_query_result_size(const zvec_db_query_result_t* result);
zvec_status_t zvec_db_query_result_get_doc(
    const zvec_db_query_result_t* result,
    size_t idx,
    zvec_db_doc_t** out_doc);
void zvec_db_query_result_destroy(zvec_db_query_result_t* result);

// Group-by query
typedef struct zvec_db_group_query zvec_db_group_query_t;
typedef struct zvec_db_group_result zvec_db_group_result_t;

typedef enum {
  ZVEC_DB_LOG_DEBUG = 0,
  ZVEC_DB_LOG_INFO = 1,
  ZVEC_DB_LOG_WARN = 2,
  ZVEC_DB_LOG_ERROR = 3,
  ZVEC_DB_LOG_FATAL = 4
} zvec_db_log_level_t;

typedef struct zvec_db_config zvec_db_config_t;

zvec_status_t zvec_db_config_create(zvec_db_config_t** out_config);
void zvec_db_config_destroy(zvec_db_config_t* config);

zvec_status_t zvec_db_config_set_memory_limit_bytes(
    zvec_db_config_t* config,
    uint64_t value);
zvec_status_t zvec_db_config_set_query_thread_count(
    zvec_db_config_t* config,
    uint32_t value);
zvec_status_t zvec_db_config_set_optimize_thread_count(
    zvec_db_config_t* config,
    uint32_t value);
zvec_status_t zvec_db_config_set_invert_to_forward_scan_ratio(
    zvec_db_config_t* config,
    float value);
zvec_status_t zvec_db_config_set_brute_force_by_keys_ratio(
    zvec_db_config_t* config,
    float value);

zvec_status_t zvec_db_config_set_console_logger(
    zvec_db_config_t* config,
    zvec_db_log_level_t level);
zvec_status_t zvec_db_config_set_file_logger(
    zvec_db_config_t* config,
    zvec_db_log_level_t level,
    const char* dir,
    const char* basename,
    uint32_t file_size_mb,
    uint32_t overdue_days);

zvec_status_t zvec_db_global_config_init(
    const zvec_db_config_t* config);

zvec_status_t zvec_db_group_query_create(
    const char* field_name,
    const char* group_by_field,
    zvec_db_group_query_t** out_query);
void zvec_db_group_query_destroy(zvec_db_group_query_t* query);
zvec_status_t zvec_db_group_query_set_group_count(
    zvec_db_group_query_t* query, uint32_t group_count);
zvec_status_t zvec_db_group_query_set_group_topk(
    zvec_db_group_query_t* query, uint32_t group_topk);
zvec_status_t zvec_db_group_query_set_include_vector(
    zvec_db_group_query_t* query, int include_vector);
zvec_status_t zvec_db_group_query_set_filter(
    zvec_db_group_query_t* query, const char* filter);
zvec_status_t zvec_db_group_query_set_output_fields(
    zvec_db_group_query_t* query, const char** fields, size_t count);
zvec_status_t zvec_db_group_query_set_dense_vector_bytes(
    zvec_db_group_query_t* query, const void* data, size_t bytes);
zvec_status_t zvec_db_group_query_set_sparse_vector(
    zvec_db_group_query_t* query,
    const uint32_t* indices,
    const void* values,
    size_t count,
    size_t value_bytes);
zvec_status_t zvec_db_group_query_set_params(
    zvec_db_group_query_t* query,
    const zvec_db_query_params_t* params);

zvec_status_t zvec_db_collection_groupby_query(
    zvec_db_collection_t* collection,
    const zvec_db_group_query_t* query,
    zvec_db_group_result_t** out_result);

size_t zvec_db_group_result_group_count(
    const zvec_db_group_result_t* result);
zvec_status_t zvec_db_group_result_get_group_value(
    const zvec_db_group_result_t* result,
    size_t group_idx,
    char** out_string);
size_t zvec_db_group_result_group_doc_count(
    const zvec_db_group_result_t* result,
    size_t group_idx);
zvec_status_t zvec_db_group_result_get_doc(
    const zvec_db_group_result_t* result,
    size_t group_idx,
    size_t doc_idx,
    zvec_db_doc_t** out_doc);
void zvec_db_group_result_destroy(zvec_db_group_result_t* result);

// -----------------------
// Ailego interface (C API)
// -----------------------

int zvec_ailego_string_starts_with(const char* ref, const char* prefix);
int zvec_ailego_string_ends_with(const char* ref, const char* suffix);
int zvec_ailego_string_compare_ignore_case(const char* a, const char* b);

zvec_status_t zvec_ailego_string_copy_trim(
    const char* input,
    char** out_string);
zvec_status_t zvec_ailego_string_copy_left_trim(
    const char* input,
    char** out_string);
zvec_status_t zvec_ailego_string_copy_right_trim(
    const char* input,
    char** out_string);

zvec_status_t zvec_ailego_string_split(
    const char* input,
    const char* delim,
    char*** out_items,
    size_t* out_count);
void zvec_ailego_string_split_free(char** items, size_t count);

#ifdef __cplusplus
}
#endif

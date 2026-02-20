package main

/*
#cgo LDFLAGS: -L/home/Adeel/development/agentic/zvec/build/lib -lzvec_all  -lstdc++ -lm -ldl -lpthread
#include "/home/Adeel/development/agentic/zvec/src/include/zvec_c.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// zvec_status_t zvec_db_doc_create(zvec_db_doc_t** out_doc);
// void zvec_db_doc_destroy(zvec_db_doc_t* doc);
// zvec_status_t zvec_db_doc_set_pk(zvec_db_doc_t* doc, const char* pk);
// zvec_status_t zvec_db_doc_set_doc_id(zvec_db_doc_t* doc, uint64_t doc_id);
// zvec_status_t zvec_db_doc_set_score(zvec_db_doc_t* doc, float score);
// zvec_status_t zvec_db_doc_set_operator(zvec_db_doc_t* doc, zvec_db_operator_t op);
func createDoc(docId uint64, schema *C.zvec_db_schema_t, pk string) *C.zvec_db_doc_t {
	var doc *C.zvec_db_doc_t

	_ = C.zvec_db_doc_create(&doc)
	if pk == "" {
		pk = fmt.Sprintf("pk_%d", docId)
	}
	cPk := C.CString(pk)
	//defer C.free(unsafe.Pointer(cPk))
	C.zvec_db_doc_set_pk(doc, cPk)

	var fieldNames **C.char
	var outSize C.size_t

	_ = C.zvec_db_schema_field_names(schema, &fieldNames, &outSize)
	//fmt.Println("Size of Field Names: ", outSize)
	allFields := make([]string, 0)
	for i := 0; i < int(outSize); i++ {
		fieldName := C.GoString(*(**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(fieldNames)) + uintptr(i)*unsafe.Sizeof(*fieldNames))))
		allFields = append(allFields, fieldName)
	}
	//zvec_status_t zvec_db_schema_get_field_by_name(
	//const zvec_db_schema_t* schema,
	//const char* name,
	//zvec_db_field_t** out_field);
	for _, fieldName := range allFields {
		var field *C.zvec_db_field_t
		_ = C.zvec_db_schema_get_field_by_name(schema, C.CString(fieldName), &field)
		//fmt.Println("Field Name: ", fieldName)

		//zvec_status_t zvec_db_field_get_data_type(
		//const zvec_db_field_t* field,
		//zvec_db_data_type_t* out_type);

		var dataType C.zvec_db_data_type_t
		_ = C.zvec_db_field_get_data_type(field, &dataType)
		//fmt.Println("Data Type: ", dataType)

		//switch dataType {
		//case C.ZVEC_DB_DT_INT64:
		//	//fmt.Println("Setting Int64")
		//	_ = C.zvec_db_doc_set_int64(doc, C.CString(fieldName), C.int64_t(docId))
		//case C.ZVEC_DB_DT_VECTOR_FP32:
		//	vector := make([]C.float, 128)
		//	for i := range vector {
		//		vector[i] = C.float(float32(docId) + float32(i))
		//	}
		//	_ = C.zvec_db_doc_set_vector_fp32(doc, C.CString(fieldName), &vector[0], 128)
		//case C.ZVEC_DB_DT_SPARSE_VECTOR_FP32:
		//	values := []C.float{1.0, 2.0}
		//	indices := []C.uint32_t{0, 1}
		//	_ = C.zvec_db_doc_set_sparse_vector_fp32(doc, C.CString(fieldName), &values[0], &indices[0], 1)
		//case C.ZVEC_DB_DT_STRING:
		//	//fmt.Println("Setting String")
		//	_ = C.zvec_db_doc_set_string(doc, C.CString(fieldName), C.CString(fmt.Sprintf("value_%d", docId)))
		//case C.ZVEC_DB_DT_FLOAT:
		//	//fmt.Println("Setting Float")
		//	floatValue := float32(docId)
		//	_ = C.zvec_db_doc_set_float(doc, C.CString(fieldName), C.float(floatValue))
		//case C.ZVEC_DB_DT_BINARY:
		//	fallthrough
		//default:
		//	fmt.Println("Unsupported data type: ", dataType)
		//	continue
		//}

		switch dataType {
		case C.ZVEC_DB_DT_INT64:
			_ = C.zvec_db_doc_set_int64(doc, C.CString(fieldName), C.int64_t(docId))
		case C.ZVEC_DB_DT_VECTOR_FP32:
			vector := make([]C.float, 128)
			for i := range vector {
				vector[i] = C.float(float32(docId) + 0.1) // Match C++: all elements the same value
			}
			status := C.zvec_db_doc_set_vector_fp32(doc, C.CString(fieldName), &vector[0], C.size_t(128))
			if status != C.ZVEC_STATUS_OK {
				fmt.Println("failed to create dense:", C.GoString(C.zvec_last_error()))
			} else {
				fmt.Println("successfully created z vector")
			}
		case C.ZVEC_DB_DT_SPARSE_VECTOR_FP32:
			nnz := 100
			values := make([]C.float, nnz)
			indices := make([]C.uint32_t, nnz)
			for i := 0; i < nnz; i++ {
				indices[i] = C.uint32_t(i)
				values[i] = C.float(float32(docId) + 0.1)
			}
			_ = C.zvec_db_doc_set_sparse_vector_fp32(doc, C.CString(fieldName), &indices[0], &values[0], C.size_t(nnz)) // Order: values, indices
		case C.ZVEC_DB_DT_STRING:
			_ = C.zvec_db_doc_set_string(doc, C.CString(fieldName), C.CString(fmt.Sprintf("value_%d", docId)))
		case C.ZVEC_DB_DT_FLOAT:
			floatValue := float32(docId)
			_ = C.zvec_db_doc_set_float(doc, C.CString(fieldName), C.float(floatValue))
		case C.ZVEC_DB_DT_BINARY:
			fallthrough
		default:
			fmt.Println("Unsupported data type: ", dataType)
		}

	}

	return doc
}

/*
Doc create_doc(const uint64_t doc_id, const CollectionSchema &schema,
               std::string pk = "") {
  Doc new_doc;
  if (pk.empty()) {
    pk = "pk_" + std::to_string(doc_id);
  }
  new_doc.set_pk(pk);

  for (auto &field : schema.fields()) {
    switch (field->data_type()) {
      case DataType::BINARY: {
        std::string binary_str("binary_" + std::to_string(doc_id));
        new_doc.set<std::string>(field->name(), binary_str);
        break;
      }
      case DataType::BOOL:
        new_doc.set<bool>(field->name(), doc_id % 10 == 0);
        break;
      case DataType::INT32:
        new_doc.set<int32_t>(field->name(), (int32_t)doc_id);
        break;
      case DataType::INT64:
        new_doc.set<int64_t>(field->name(), (int64_t)doc_id);
        break;
      case DataType::UINT32:
        new_doc.set<uint32_t>(field->name(), (uint32_t)doc_id);
        break;
      case DataType::UINT64:
        new_doc.set<uint64_t>(field->name(), (uint64_t)doc_id);
        break;
      case DataType::FLOAT:
        new_doc.set<float>(field->name(), (float)doc_id);
        break;
      case DataType::DOUBLE:
        new_doc.set<double>(field->name(), (double)doc_id);
        break;
      case DataType::STRING:
        new_doc.set<std::string>(field->name(),
                                 "value_" + std::to_string(doc_id));
        break;
      case DataType::ARRAY_BINARY: {
        std::vector<std::string> bin_vec;
        for (size_t i = 0; i < (doc_id % 10); i++) {
          bin_vec.push_back("bin_" + std::to_string(i));
        }
        new_doc.set<std::vector<std::string>>(field->name(), bin_vec);
        break;
      }
      case DataType::ARRAY_BOOL:
        new_doc.set<std::vector<bool>>(field->name(),
                                       std::vector<bool>(10, doc_id % 10 == 0));
        break;
      case DataType::ARRAY_INT32:
        new_doc.set<std::vector<int32_t>>(
            field->name(), std::vector<int32_t>(10, (int32_t)doc_id));
        break;
      case DataType::ARRAY_INT64:
        new_doc.set<std::vector<int64_t>>(
            field->name(), std::vector<int64_t>(10, (int64_t)doc_id));
        break;
      case DataType::ARRAY_UINT32:
        new_doc.set<std::vector<uint32_t>>(
            field->name(), std::vector<uint32_t>(10, (uint32_t)doc_id));
        break;
      case DataType::ARRAY_UINT64:
        new_doc.set<std::vector<uint64_t>>(
            field->name(), std::vector<uint64_t>(10, (uint64_t)doc_id));
        break;
      case DataType::ARRAY_FLOAT:
        new_doc.set<std::vector<float>>(field->name(),
                                        std::vector<float>(10, (float)doc_id));
        break;
      case DataType::ARRAY_DOUBLE:
        new_doc.set<std::vector<double>>(
            field->name(), std::vector<double>(10, (double)doc_id));
        break;
      case DataType::ARRAY_STRING:
        new_doc.set<std::vector<std::string>>(
            field->name(),
            std::vector<std::string>(10, "value_" + std::to_string(doc_id)));
        break;
      case DataType::VECTOR_BINARY32:
        new_doc.set<std::vector<uint32_t>>(
            field->name(),
            std::vector<uint32_t>(field->dimension(), uint32_t(doc_id + 0.1)));
        break;
      case DataType::VECTOR_BINARY64:
        new_doc.set<std::vector<uint64_t>>(
            field->name(),
            std::vector<uint64_t>(field->dimension(), uint64_t(doc_id + 0.1)));
        break;
      case DataType::VECTOR_FP32:
        new_doc.set<std::vector<float>>(
            field->name(),
            std::vector<float>(field->dimension(), float(doc_id + 0.1)));
        break;
      case DataType::VECTOR_FP64:
        new_doc.set<std::vector<double>>(
            field->name(),
            std::vector<double>(field->dimension(), double(doc_id + 0.1)));
        break;
      case DataType::VECTOR_FP16:
        new_doc.set<std::vector<float16_t>>(
            field->name(), std::vector<float16_t>(
                               field->dimension(),
                               static_cast<float16_t>(float(doc_id + 0.1))));
        break;
      case DataType::VECTOR_INT8:
        new_doc.set<std::vector<int8_t>>(
            field->name(),
            std::vector<int8_t>(field->dimension(), (int8_t)doc_id));
        break;
      case DataType::VECTOR_INT16:
        new_doc.set<std::vector<int16_t>>(
            field->name(),
            std::vector<int16_t>(field->dimension(), (int16_t)doc_id));
        break;
      case DataType::SPARSE_VECTOR_FP16: {
        std::vector<uint32_t> indices;
        std::vector<float16_t> values;
        for (uint32_t i = 0; i < 100; i++) {
          indices.push_back(i);
          values.push_back(float16_t(float(doc_id + 0.1)));
        }
        std::pair<std::vector<uint32_t>, std::vector<float16_t>>
            sparse_float_vec;
        sparse_float_vec.first = indices;
        sparse_float_vec.second = values;
        new_doc.set<std::pair<std::vector<uint32_t>, std::vector<float16_t>>>(
            field->name(), sparse_float_vec);
        break;
      }
      case DataType::SPARSE_VECTOR_FP32: {
        std::vector<uint32_t> indices;
        std::vector<float> values;
        for (uint32_t i = 0; i < 100; i++) {
          indices.push_back(i);
          values.push_back(float(doc_id + 0.1));
        }
        std::pair<std::vector<uint32_t>, std::vector<float>> sparse_float_vec;
        sparse_float_vec.first = indices;
        sparse_float_vec.second = values;
        new_doc.set<std::pair<std::vector<uint32_t>, std::vector<float>>>(
            field->name(), sparse_float_vec);
        break;
      }
      default:
        std::cout << "Unsupported data type: " << field->name() << std::endl;
        throw std::runtime_error("Unsupported vector data type");
    }
  }

  return new_doc;
}

*/

//make zvec_all_static_archive -j 4
// zvec_status_t zvec_db_schema_create(const char* name, zvec_db_schema_t** out);
/*
zvec_status_t zvec_db_collection_create_and_open(
    const char* path,
    const zvec_db_schema_t* schema,
    int read_only,
    int enable_mmap,
    uint32_t max_buffer_size,
    zvec_db_collection_t** out_collection);
*/

/*
zvec_status_t zvec_db_schema_create(const char* name, zvec_db_schema_t** out);
zvec_status_t zvec_db_schema_set_max_docs_per_segment(
    zvec_db_schema_t* schema,
    uint64_t max_docs);
*/

//CollectionSchema::Ptr create_schema() {
//auto schema = std::make_shared<CollectionSchema>("demo");
//schema->set_max_doc_count_per_segment(1000);
//
//schema->add_field(std::make_shared<FieldSchema>(
//"id", DataType::INT64, false, std::make_shared<InvertIndexParams>(true)));
//schema->add_field(std::make_shared<FieldSchema>(
//"name", DataType::STRING, false,
//std::make_shared<InvertIndexParams>(false)));
//schema->add_field(
//std::make_shared<FieldSchema>("weight", DataType::FLOAT, true));
//
//schema->add_field(std::make_shared<FieldSchema>(
//"dense", DataType::VECTOR_FP32, 128, false,
//std::make_shared<HnswIndexParams>(MetricType::IP)));
//schema->add_field(std::make_shared<FieldSchema>(
//"sparse", DataType::SPARSE_VECTOR_FP32, 0, false,
//std::make_shared<HnswIndexParams>(MetricType::IP)));
//
//return schema;
//}

//zvec_status_t zvec_db_schema_add_field(
//zvec_db_schema_t* schema,
//const zvec_db_field_t* field);

/*
zvec_status_t zvec_db_field_create_scalar(

	const char* name,
	zvec_db_data_type_t data_type,
	int nullable,
	const zvec_db_index_params_t* index_params,
	zvec_db_field_t** out_field);
*/
func createSchema() *C.zvec_db_schema_t {
	name := C.CString("demo")
	//defer C.free(unsafe.Pointer(name))

	var schema *C.zvec_db_schema_t

	_ = C.zvec_db_schema_create(name, &schema)
	//defer C.zvec_db_schema_destroy(schema)

	_ = C.zvec_db_schema_set_max_docs_per_segment(schema, 1000)

	var field *C.zvec_db_field_t
	//defer C.zvec_db_field_destroy(field)
	var invertParam *C.zvec_db_index_params_t
	_ = C.zvec_db_index_params_create_invert(C.int(1), C.int(0), &invertParam)
	_ = C.zvec_db_field_create_scalar(C.CString("id"), C.ZVEC_DB_DT_INT64, C.int(0), invertParam, &field)
	_ = C.zvec_db_schema_add_field(schema, field)

	var field2 *C.zvec_db_field_t
	//defer C.zvec_db_field_destroy(field2)
	var invertParam2 *C.zvec_db_index_params_t
	_ = C.zvec_db_index_params_create_invert(C.int(0), C.int(0), &invertParam2)
	_ = C.zvec_db_field_create_scalar(C.CString("name"), C.ZVEC_DB_DT_STRING, C.int(0), invertParam2, &field2)
	_ = C.zvec_db_schema_add_field(schema, field2)

	var field3 *C.zvec_db_field_t
	//defer C.zvec_db_field_destroy(field3)
	var invertParam3 *C.zvec_db_index_params_t
	_ = C.zvec_db_index_params_create_invert(C.int(0), C.int(1), &invertParam3)
	_ = C.zvec_db_field_create_scalar(C.CString("weight"), C.ZVEC_DB_DT_FLOAT, C.int(1), invertParam3, &field3)
	_ = C.zvec_db_schema_add_field(schema, field3)

	var field4 *C.zvec_db_field_t
	//defer C.zvec_db_field_destroy(field4)
	var hnswParam *C.zvec_db_index_params_t
	//defer C.zvec_db_index_params_destroy(hnswParam)

	var metric C.zvec_db_metric_type_t
	metric = C.ZVEC_DB_METRIC_IP
	status := C.zvec_db_index_params_create_hnsw(metric, C.ZVEC_CORE_DEFAULT_HNSW_NEIGHBOR_CNT, C.ZVEC_CORE_DEFAULT_HNSW_EF_CONSTRUCTION, C.ZVEC_DB_QUANTIZE_NONE, &hnswParam)
	if status != C.ZVEC_STATUS_OK {
		fmt.Println("373:", C.GoString(C.zvec_last_error()))
	} else {
		fmt.Println("375 hnsw created successfully")
	}
	_ = C.zvec_db_field_create_vector(C.CString("dense"), C.ZVEC_DB_DT_VECTOR_FP32, 128, C.int(0), hnswParam, &field4)
	_ = C.zvec_db_schema_add_field(schema, field4)

	var field5 *C.zvec_db_field_t
	//defer C.zvec_db_field_destroy(field5)
	var hnswParam2 *C.zvec_db_index_params_t
	//defer C.zvec_db_index_params_destroy(hnswParam2)
	_ = C.zvec_db_index_params_create_hnsw(metric, C.ZVEC_CORE_DEFAULT_HNSW_NEIGHBOR_CNT, C.ZVEC_CORE_DEFAULT_HNSW_EF_CONSTRUCTION, C.ZVEC_DB_QUANTIZE_NONE, &hnswParam2)
	_ = C.zvec_db_field_create_sparse_vector(C.CString("sparse"), C.ZVEC_DB_DT_SPARSE_VECTOR_FP32, 0, C.int(0), hnswParam2, &field5)
	_ = C.zvec_db_schema_add_field(schema, field5)

	return schema
}

func main() {
	path := "./demo"
	cPath := C.CString(path)
	//defer C.free(unsafe.Pointer(cPath))
	fmt.Println(C.GoString(cPath))

	rmCmd := "rm -rf " + path
	cRmCmd := C.CString(rmCmd)
	//defer C.free(unsafe.Pointer(cRmCmd))
	fmt.Println(C.GoString(cRmCmd))

	C.system(cRmCmd)

	schema := createSchema()
	//defer C.zvec_db_schema_destroy(schema)

	var collection *C.zvec_db_collection_t
	status := C.zvec_db_collection_create_and_open(cPath, schema, C.int(0), C.int(1), C.uint32_t(10*1024*1024), &collection)
	if status != C.ZVEC_STATUS_OK {
		fmt.Println("Create collection failed:", C.GoString(C.zvec_last_error()))
		return
	}
	//defer C.zvec_db_collection_destroy(collection)
	//defer C.zvec_db_collection_destroy(collection)

	/*
		zvec_status_t zvec_db_collection_stats(
		    zvec_db_collection_t* collection,
		    char** out_string);
	*/
	var stats *C.char
	C.zvec_db_collection_stats(collection, &stats)
	fmt.Println(C.GoString(stats))

	// Insert a single document
	{
		fmt.Println("Creating and inserting one doc")

		// Create one document
		doc := createDoc(0, schema, "")
		//defer C.zvec_db_doc_destroy(doc) // Ensure cleanup

		// Insert the single document
		status := C.zvec_db_collection_insert(collection, &doc, C.size_t(1))
		if status != C.ZVEC_STATUS_OK {
			fmt.Println("Insert failed:", C.GoString(C.zvec_last_error()))
		} else {
			fmt.Println("Inserted one document successfully")
		}

		// Print stats after insert
		var stats *C.char
		//defer C.free(unsafe.Pointer(stats)) // Free the stats string
		C.zvec_db_collection_stats(collection, &stats)
		fmt.Println("After insert stats:", C.GoString(stats))
	}

	{
		//zvec_status_t zvec_db_collection_optimize(zvec_db_collection_t* collection);
		status = C.zvec_db_collection_optimize(collection)
		if status != C.ZVEC_STATUS_OK {
			fmt.Println("Optimization failed:", C.GoString(C.zvec_last_error()))
			return
		}
		var stats *C.char
		C.zvec_db_collection_stats(collection, &stats)

		fmt.Println("After Optimizing Stats:", C.GoString(stats))
	}
	{
		var vectorQuery *C.zvec_db_vector_query_t
		_ = C.zvec_db_vector_query_create(C.CString("dense"), &vectorQuery)
		//defer C.zvec_db_vector_query_destroy(vectorQuery)
		_ = C.zvec_db_vector_query_set_topk(vectorQuery, C.int(10))
		_ = C.zvec_db_vector_query_set_include_vector(vectorQuery, C.int(1))

		queryVec := make([]C.float, 128)

		for i := range queryVec {
			queryVec[i] = C.float(0.1)
		}

		_ = C.zvec_db_vector_query_set_dense_vector_fp32(
			vectorQuery,
			&queryVec[0],
			C.size_t(len(queryVec)),
		)

		var result *C.zvec_db_query_result_t
		_ = C.zvec_db_collection_vector_query(collection, vectorQuery, &result)

		count := C.zvec_db_query_result_size(result)
		fmt.Printf("query result: doc_count[%d]\n", count)

		if count == 0 {
			fmt.Println("first doc: <none>")
		} else {
			var doc *C.zvec_db_doc_t
			C.zvec_db_query_result_get_doc(result, 0, &doc)
			var docStr *C.char
			C.zvec_db_doc_to_detail_string(doc, &docStr)
			fmt.Println("first doc:", C.GoString(docStr))
		}

	}
	/*
		{


		    VectorQuery query;
		    query.topk_ = 10;
		    query.field_name_ = "dense";
		    query.include_vector_ = true;
		    std::vector<float> query_vector = std::vector<float>(128, 0.1);
		    query.query_vector_.assign((char *)query_vector.data(),
		                               query_vector.size() * sizeof(float));
		    auto res = coll->Query(query);
		    if (!res.has_value()) {
		      std::cout << res.error().message() << std::endl;
		      return -1;
		    }
		    std::cout << "query result: doc_count[" << res.value().size() << "]"
		              << std::endl;
		    if (res.value().empty() || !res.value()[0]) {
		      std::cout << "first doc: <none>" << std::endl;
		    } else {
		      std::cout << "first doc: " << res.value()[0]->to_detail_string()
		                << std::endl;
		    }
		  }
	*/

}

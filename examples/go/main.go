package main

/*
#cgo CFLAGS: -I../../src/include
#cgo LDFLAGS: -L../../cmake-build-release/lib -lzvec_all -lstdc++ -lm -ldl -lpthread
#include "zvec_c.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

func check(status C.zvec_status_t) {
	if status != C.ZVEC_STATUS_OK {
		msg := C.GoString(C.zvec_last_error())
		panic(msg)
	}
}

func main() {
	var param *C.zvec_core_param_t
	check(C.zvec_core_param_create_flat(
		C.ZVEC_CORE_METRIC_INNER_PRODUCT,
		C.ZVEC_CORE_DT_FP32,
		4,
		0,
		&param,
	))
	defer C.zvec_core_param_destroy(param)

	var index *C.zvec_core_index_t
	check(C.zvec_core_index_create(param, &index))
	defer C.zvec_core_index_destroy(index)

	path := C.CString("./zvec_core_go")
	defer C.free(unsafe.Pointer(path))
	check(C.zvec_core_index_open(index, path, C.ZVEC_CORE_STORAGE_MMAP, 1, 0))

	vec := []float32{0.1, 0.2, 0.3, 0.4}
	check(C.zvec_core_index_add_dense(
		index,
		unsafe.Pointer(&vec[0]),
		C.size_t(len(vec)),
		1,
	))

	var qparam *C.zvec_core_query_param_t
	check(C.zvec_core_query_param_create_flat(3, 0, 0, 0, &qparam))
	defer C.zvec_core_query_param_destroy(qparam)

	query := []float32{0.1, 0.2, 0.3, 0.4}
	var result *C.zvec_core_search_result_t
	check(C.zvec_core_index_search_dense(
		index,
		unsafe.Pointer(&query[0]),
		C.size_t(len(query)),
		qparam,
		&result,
	))
	defer C.zvec_core_search_result_destroy(result)

	n := int(C.zvec_core_search_result_size(result))
	fmt.Printf("results: %d\n", n)
	for i := 0; i < n; i++ {
		var key C.uint64_t
		var score C.float
		check(C.zvec_core_search_result_get(result, C.size_t(i), &key, &score))
		fmt.Printf("doc_id=%d score=%.4f\n", uint64(key), float32(score))
	}
}

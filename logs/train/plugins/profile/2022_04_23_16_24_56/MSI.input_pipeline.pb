	?p=
???@?p=
???@!?p=
???@	8?"ql?8?"ql?!8?"ql?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?p=
???@|a2U0*??AGx???@Y???V?/??*	3333??@2]
&Iterator::Model::Prefetch::MapAndBatch?O??U0@!0??]??X@)?O??U0@10??]??X@:Preprocessing2k
3Iterator::Model::Prefetch::MapAndBatch::TensorSlice?Y?8??m??!L?-?? ??)Y?8??m??1L?-?? ??:Preprocessing2F
Iterator::Modelm???{???!??-T????)p_?Q??1?K`???:Preprocessing2P
Iterator::Model::Prefetch???_vO~?!??ϗ??)???_vO~?1??ϗ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no98?"ql?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	|a2U0*??|a2U0*??!|a2U0*??      ??!       "      ??!       *      ??!       2	Gx???@Gx???@!Gx???@:      ??!       B      ??!       J	???V?/?????V?/??!???V?/??R      ??!       Z	???V?/?????V?/??!???V?/??JCPU_ONLYY8?"ql?b 
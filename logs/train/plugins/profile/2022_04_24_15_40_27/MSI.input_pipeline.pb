	&S??%?@&S??%?@!&S??%?@		??{??l?	??{??l?!	??{??l?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$&S??%?@?G?z???A?????$?@Y?^)?Ǫ?*	gffffv`@2j
3Iterator::Model::Prefetch::MapAndBatch::TensorSlice
o?ŏ1??!???Ż2Q@)o?ŏ1??1???Ż2Q@:Preprocessing2F
Iterator::Model?q??????!t?F??7@)?HP???1ά;???2@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch??ׁsF??!??Ɋ~@)??ׁsF??1??Ɋ~@:Preprocessing2P
Iterator::Model::Prefetch_?Q?{?!?O?*?@)_?Q?{?1?O?*?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??{??l?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?G?z????G?z???!?G?z???      ??!       "      ??!       *      ??!       2	?????$?@?????$?@!?????$?@:      ??!       B      ??!       J	?^)?Ǫ??^)?Ǫ?!?^)?Ǫ?R      ??!       Z	?^)?Ǫ??^)?Ǫ?!?^)?Ǫ?JCPU_ONLYY??{??l?b 
	Qk?w???@Qk?w???@!Qk?w???@	?x??x??x??x?!?x??x?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Qk?w???@???&??A??JY???@Yo?ŏ1??*	????Y?@2]
&Iterator::Model::Prefetch::MapAndBatch?s??9@!Z?W?V?X@)?s??9@1Z?W?V?X@:Preprocessing2k
3Iterator::Model::Prefetch::MapAndBatch::TensorSlice??@??ǘ??!)?.?????)?@??ǘ??1)?.?????:Preprocessing2F
Iterator::ModelV}??b??!z?݂ڸ?)?j+??ݓ?1??<??\??:Preprocessing2P
Iterator::Model::Prefetch?I+?v?!e?&'???)?I+?v?1e?&'???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?x??x?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???&?????&??!???&??      ??!       "      ??!       *      ??!       2	??JY???@??JY???@!??JY???@:      ??!       B      ??!       J	o?ŏ1??o?ŏ1??!o?ŏ1??R      ??!       Z	o?ŏ1??o?ŏ1??!o?ŏ1??JCPU_ONLYY?x??x?b 
	o??7&?@o??7&?@!o??7&?@	??V??e>???V??e>?!??V??e>?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$o??7&?@????????A?7??&?@Y??d?`T??*	     ??@2j
3Iterator::Model::Prefetch::MapAndBatch::TensorSlice ??v????!?O??rN@)??v????1?O??rN@:Preprocessing2F
Iterator::Modelw??/???!?2E?u?B@)??ܵ?|??1!!t???B@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch?? ?rh??!?O!????)?? ?rh??1?O!????:Preprocessing2P
Iterator::Model::Prefetch??0?*x?!?vD4\%??)??0?*x?1?vD4\%??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??V??e>?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????????????????!????????      ??!       "      ??!       *      ??!       2	?7??&?@?7??&?@!?7??&?@:      ??!       B      ??!       J	??d?`T????d?`T??!??d?`T??R      ??!       Z	??d?`T????d?`T??!??d?`T??JCPU_ONLYY??V??e>?b 
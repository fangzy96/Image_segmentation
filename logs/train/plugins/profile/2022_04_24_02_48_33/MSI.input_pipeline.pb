	lxz?읤@lxz?읤@!lxz?읤@	?H???8b??H???8b?!?H???8b?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$lxz?읤@?^)???A?4??@Y??(???*	fffff?^@2j
3Iterator::Model::Prefetch::MapAndBatch::TensorSlice
]?Fx??!嬺???O@)]?Fx??1嬺???O@:Preprocessing2F
Iterator::Model?:pΈ??!_???=@)y?&1???1?u????6@:Preprocessing2P
Iterator::Model::Prefetch	?^)ˀ?!j?Q???@)	?^)ˀ?1j?Q???@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch??ǘ????!]?~jm@)??ǘ????1]?~jm@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?H???8b?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?^)????^)???!?^)???      ??!       "      ??!       *      ??!       2	?4??@?4??@!?4??@:      ??!       B      ??!       J	??(?????(???!??(???R      ??!       Z	??(?????(???!??(???JCPU_ONLYY?H???8b?b 
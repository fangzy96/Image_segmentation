	?_vO??@?_vO??@!?_vO??@	??W2?Y???W2?Y?!??W2?Y?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?_vO??@?rh??|??A?V~??@Y/n????*	fffff&`@2j
3Iterator::Model::Prefetch::MapAndBatch::TensorSlice
?ݓ??Z??!??&??AM@)?ݓ??Z??1??&??AM@:Preprocessing2F
Iterator::Model?z6?>??!?[?3ՑA@)r??????1???=f;@:Preprocessing2P
Iterator::Model::Prefetch{?G?z??!?+T??@){?G?z??1?+T??@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch	?^)ˀ?!?p?:c@)	?^)ˀ?1?p?:c@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??W2?Y?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?rh??|???rh??|??!?rh??|??      ??!       "      ??!       *      ??!       2	?V~??@?V~??@!?V~??@:      ??!       B      ??!       J	/n????/n????!/n????R      ??!       Z	/n????/n????!/n????JCPU_ONLYY??W2?Y?b 
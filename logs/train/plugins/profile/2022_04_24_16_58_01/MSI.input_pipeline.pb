	Y?8?E??@Y?8?E??@!Y?8?E??@	??HF?|???HF?|?!??HF?|?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Y?8?E??@Z??ڊ???Ash??|??@YNё\?C??*	??????`@2j
3Iterator::Model::Prefetch::MapAndBatch::TensorSliceV-?????!???M?Q@)V-?????1???M?Q@:Preprocessing2F
Iterator::Modelm???{???!?????8@)??q????1?#n.4@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatchS?!?uq{?!*1 ?@)S?!?uq{?1*1 ?@:Preprocessing2P
Iterator::Model::Prefetcha??+ey?!??)kʚ@)a??+ey?1??)kʚ@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??HF?|?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Z??ڊ???Z??ڊ???!Z??ڊ???      ??!       "      ??!       *      ??!       2	sh??|??@sh??|??@!sh??|??@:      ??!       B      ??!       J	Nё\?C??Nё\?C??!Nё\?C??R      ??!       Z	Nё\?C??Nё\?C??!Nё\?C??JCPU_ONLYY??HF?|?b 
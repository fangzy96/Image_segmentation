	2U0*)I?@2U0*)I?@!2U0*)I?@	?j?`B?z??j?`B?z?!?j?`B?z?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$2U0*)I?@i o????AX9?HG?@Y(~??k	??*	??????R@2j
3Iterator::Model::Prefetch::MapAndBatch::TensorSlice?l??????!??Z??H@)?l??????1??Z??H@:Preprocessing2F
Iterator::Model????o??!??NBv~D@)Ǻ?????1*J?#?=@:Preprocessing2P
Iterator::Model::Prefetch?St$????!??s??)&@)?St$????1??s??)&@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch??H?}}?!?A[)?9#@)??H?}}?1?A[)?9#@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?j?`B?z?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	i o????i o????!i o????      ??!       "      ??!       *      ??!       2	X9?HG?@X9?HG?@!X9?HG?@:      ??!       B      ??!       J	(~??k	??(~??k	??!(~??k	??R      ??!       Z	(~??k	??(~??k	??!(~??k	??JCPU_ONLYY?j?`B?z?b 
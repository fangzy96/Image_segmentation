	?Zd[ԡ@?Zd[ԡ@!?Zd[ԡ@	R/???d?R/???d?!R/???d?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?Zd[ԡ@ ?~?:p??A|??P?ӡ@Y6?;Nё??*	?????y[@2j
3Iterator::Model::Prefetch::MapAndBatch::TensorSlice
ۊ?e????!?h#8?O@)ۊ?e????1?h#8?O@:Preprocessing2F
Iterator::Model??ǘ????!?z3?9{=@)p_?Q??1????b7@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatchŏ1w-!?!???X?@)ŏ1w-!?1???X?@:Preprocessing2P
Iterator::Model::PrefetchS?!?uq{?!?y"?b@)S?!?uq{?1?y"?b@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9R/???d?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	 ?~?:p?? ?~?:p??! ?~?:p??      ??!       "      ??!       *      ??!       2	|??P?ӡ@|??P?ӡ@!|??P?ӡ@:      ??!       B      ??!       J	6?;Nё??6?;Nё??!6?;Nё??R      ??!       Z	6?;Nё??6?;Nё??!6?;Nё??JCPU_ONLYYR/???d?b 
	?q??v?@?q??v?@!?q??v?@	??y?_????y?_??!??y?_??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?q??v?@???K7???At$??p?@Y'?Wʢ?*	     ??@2k
3Iterator::Model::Prefetch::MapAndBatch::TensorSlice??z?G???!?????X@)?z?G???1?????X@:Preprocessing2F
Iterator::ModelM??St$??!???+??)?St$????1?ߤ???:Preprocessing2P
Iterator::Model::Prefetch?~j?t?x?!![8??{??)?~j?t?x?1![8??{??:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch??_vOv?!??e?;??)??_vOv?1??e?;??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??y?_??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???K7??????K7???!???K7???      ??!       "      ??!       *      ??!       2	t$??p?@t$??p?@!t$??p?@:      ??!       B      ??!       J	'?Wʢ?'?Wʢ?!'?Wʢ?R      ??!       Z	'?Wʢ?'?Wʢ?!'?Wʢ?JCPU_ONLYY??y?_??b 
	TR'????@TR'????@!TR'????@	p?b?[f?p?b?[f?!p?b?[f?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$TR'????@??ǘ????AO??僡@YK?=?U??*	     ?a@2j
3Iterator::Model::Prefetch::MapAndBatch::TensorSlice
鷯猸?!      Q@)鷯猸?1      Q@:Preprocessing2F
Iterator::Model?ݓ??Z??!      ;@)\ A?c̝?1?$I?$?4@:Preprocessing2P
Iterator::Model::Prefetch?5?;Nс?!?m۶m?@)?5?;Nс?1?m۶m?@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch ?o_?y?!      @) ?o_?y?1      @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9p?b?[f?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??ǘ??????ǘ????!??ǘ????      ??!       "      ??!       *      ??!       2	O??僡@O??僡@!O??僡@:      ??!       B      ??!       J	K?=?U??K?=?U??!K?=?U??R      ??!       Z	K?=?U??K?=?U??!K?=?U??JCPU_ONLYYp?b?[f?b 
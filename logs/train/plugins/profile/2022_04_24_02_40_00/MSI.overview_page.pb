?	<Nёܠ?@<Nёܠ?@!<Nёܠ?@	ћ\f,?b?ћ\f,?b?!ћ\f,?b?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$<Nёܠ?@??Q???A???J??@Yp_?Q??*	     `_@2j
3Iterator::Model::Prefetch::MapAndBatch::TensorSlice
#J{?/L??!????YQ@)#J{?/L??1????YQ@:Preprocessing2F
Iterator::Model?|a2U??!EtJuk9@)Zd;?O???1?.?*?S2@:Preprocessing2P
Iterator::Model::Prefetch??y?):??!k?)?]@)??y?):??1k?)?]@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch9??v??z?!	5?핷@)9??v??z?1	5?핷@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9ћ\f,?b?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??Q?????Q???!??Q???      ??!       "      ??!       *      ??!       2	???J??@???J??@!???J??@:      ??!       B      ??!       J	p_?Q??p_?Q??!p_?Q??R      ??!       Z	p_?Q??p_?Q??!p_?Q??JCPU_ONLYYћ\f,?b?b Y      Y@q ?(????"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 
�	Z.�s�n@Z.�s�n@!Z.�s�n@	��	��8�?��	��8�?!��	��8�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Z.�s�n@�v� ݗ�?A�m�B�Qn@Y��Pj/b�?*	bX94D�@2t
<Iterator::Model::MaxIntraOpParallelism::BatchV2::TensorSlice�V�y���?!��̶J@)V�y���?1��̶J@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::BatchV2�e����?!����X@)�׻?ޫ�?1o�ea�F@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism�f�R@��?!��y�Y�X@)L��pvky?1L�c���?:Preprocessing2F
Iterator::Model/�혺��?!      Y@)�;Fzq?1���oL�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9��	��8�?I��7:��X@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�v� ݗ�?�v� ݗ�?!�v� ݗ�?      ��!       "      ��!       *      ��!       2	�m�B�Qn@�m�B�Qn@!�m�B�Qn@:      ��!       B      ��!       J	��Pj/b�?��Pj/b�?!��Pj/b�?R      ��!       Z	��Pj/b�?��Pj/b�?!��Pj/b�?b      ��!       JCPU_ONLYY��	��8�?b q��7:��X@Y      Y@q�+CX��?"�
device�Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 
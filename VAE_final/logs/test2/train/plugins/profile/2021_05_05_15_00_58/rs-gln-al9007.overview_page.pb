�	�_u��p@�_u��p@!�_u��p@	��ox��?��ox��?!��ox��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�_u��p@{m��]�?A^gE��p@YQ�l�?*	@5^�I|]@2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�ՏM�#�?!�y�U��8@)�ՏM�#�?1�y�U��8@:Preprocessing2U
Iterator::Model::ParallelMapV2�I�5�o�?!�D����2@)�I�5�o�?1�D����2@:Preprocessing2F
Iterator::Model(������?!��`ps`A@)lZ)r��?1ׄY5-0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�{��?!�)�H�"3@)���]/M�?1��-@��,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?T1��?!���G�OP@)�H��Q,�?1��8C�/#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�c���Ȥ?!ӷu��5A@)��G܆?1������"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor2 Ǟ=w?!��X�L>@)2 Ǟ=w?1��X�L>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapG�?!�MZ�BB@)����(@d?1�_I� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9��ox��?Iy؞W�X@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	{m��]�?{m��]�?!{m��]�?      ��!       "      ��!       *      ��!       2	^gE��p@^gE��p@!^gE��p@:      ��!       B      ��!       J	Q�l�?Q�l�?!Q�l�?R      ��!       Z	Q�l�?Q�l�?!Q�l�?b      ��!       JCPU_ONLYY��ox��?b qy؞W�X@Y      Y@q�ݚ�'��?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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
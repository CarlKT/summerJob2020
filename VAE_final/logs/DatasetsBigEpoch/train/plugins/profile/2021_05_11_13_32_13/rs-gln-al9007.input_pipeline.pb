	���t�<<@���t�<<@!���t�<<@	��O�
k-@��O�
k-@!��O�
k-@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$���t�<<@�r�m�B�?A=��7@YZ*oG8�@*	���M���@2s
<Iterator::Model::MaxIntraOpParallelism::BatchV2::TensorSlice@��|	\@!_����R@)��|	\@1_����R@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::BatchV2$���.@!2����X@)���v�?1L?,�8@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism5��;@!��p�N�X@)�"j��G�?1��&TSo�?:Preprocessing2F
Iterator::Model��'�B@!      Y@)
��.��{?1C@Ob�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 14.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9��O�
k-@Iv��RU@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�r�m�B�?�r�m�B�?!�r�m�B�?      ��!       "      ��!       *      ��!       2	=��7@=��7@!=��7@:      ��!       B      ��!       J	Z*oG8�@Z*oG8�@!Z*oG8�@R      ��!       Z	Z*oG8�@Z*oG8�@!Z*oG8�@b      ��!       JCPU_ONLYY��O�
k-@b qv��RU@
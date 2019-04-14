[componentInstances:cComponentManager]
instance[funcsSink].type=cCsvSink

[funcsSink:cCsvSink]
reader.dmLevel = func
append = 1
timestamp = 1
printHeader = 1
delimChar = ;
number = 0
instanceName = \cm[linename(N){}:]
filename = \cm[outputfileappend(A){output.csv}:file name of the output CSV file]

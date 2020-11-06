#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark import SparkContext, SparkConf, HiveContext
conf=SparkConf()
conf.setAppName("My app")
sc = SparkContext(conf=conf)


# # 转换操作
# 对于RDD而言，每一次转换操作都会产生不同的RDD，供给下一个“转换”使用。转换得到的RDD是惰性求值的，也就是说，整个转换过程只是记录了转换的轨迹，并不会发生真正的计算，只有遇到行动操作时，才会发生真正的计算，开始从血缘关系源头开始，进行物理的转换操作。
# 下面列出一些常见的转换操作（Transformation API）：
# * filter(func)：筛选出满足函数func的元素，并返回一个新的数据集
# * map(func)：将每个元素传递到函数func中，并将结果返回为一个新的数据集
# * flatMap(func)：与map()相似，但每个输入元素都可以映射到0或多个输出结果
# * groupByKey()：应用于(K,V)键值对的数据集时，返回一个新的(K, Iterable)形式的数据集
# * reduceByKey(func)：应用于(K,V)键值对的数据集时，返回一个新的(K, V)形式的数据集，其中的每个值是将每个key传递到函数func中进行聚合

# In[4]:


lines=sc.textFile("hdfs:///hive/zxx.db/t")
lines.collect()


# In[9]:


sc.parallelize([1,2,3,4,5]).collect()


# In[10]:


sc.parallelize([1,2,3,4,5]).map(lambda x:x+1).collect()


# In[11]:


sc.parallelize([1,2,3,4,5]).filter(lambda x:x%2==0).collect()


# In[25]:


sc.parallelize(['abc acd','ace']).flatMap(lambda x:x.split(' ')).collect()


# In[44]:


sc.parallelize([('a',1),('a',1),('b',1)]).groupByKey().collect()


# In[50]:


sc.parallelize([('a',1),('a',1),('b',2),('b',1)]).reduceByKey(lambda a,b:a+b).collect()


# # 行动操作
# 行动操作是真正触发计算的地方。Spark程序执行到行动操作时，才会执行真正的计算，从文件中加载数据，完成一次又一次转换操作，最终，完成行动操作得到结果。
# 下面列出一些常见的行动操作（Action API）：
# * count() 返回数据集中的元素个数
# * collect() 以数组的形式返回数据集中的所有元素
# * first() 返回数据集中的第一个元素
# * take(n) 以数组的形式返回数据集中的前n个元素
# * reduce(func) 通过函数func（输入两个参数并返回一个值）聚合数据集中的元素
# * foreach(func) 将数据集中的每个元素传递到函数func中运行*

# In[112]:


sc.parallelize([1,2,3,4,5]).count()
sc.parallelize([1,2,3,4,5]).collect()
sc.parallelize([1,2,3,4,5]).sum()


# In[49]:


sc.parallelize([1,2,3,4,5]).first()
sc.parallelize([1,2,3,4,5]).take(3)


# In[66]:


sc.parallelize([1,2,3,4,5]).foreach(print)


# # 综合实例

# In[81]:


sc.parallelize(['abc','acd','ace']).flatMap(lambda x:x).map(lambda x:(x,1)).reduceByKey(lambda a,b:a+b).collect()


# # 键值对RDD
# ### 创建方式

# In[82]:


sc.parallelize([1,2,3,4,5]).map(lambda x:(x,1)).collect()


# # 键值对RDD
# ### 常用的键值对转换操作包括
# reduceByKey() 根据键值，将相同key的value求和
# groupByKey() 根据键值，将相同key的value组成一个tuple
# sortByKey(True/False) 返回一个根据键排序的RDD
# sortBy(func) 根据value值进行排序
# mapValues(func) 对键值对RDD中的每个value都应用一个函数，但是，key不会发生变化
# join() join的类型也和关系数据库中的join一样，包括内连接(join)、左外连接(leftOuterJoin)、右外连接(rightOuterJoin)等。最常用的情形是内连接，所以，join就表示内连接。
# 对于内连接，对于给定的两个输入数据集(K,V1)和(K,V2)，只有在两个数据集中都存在的key才会被输出，最终得到一个(K,(V1,V2))类型的数据集。
# cogroup()
# keys()
# values()

# In[94]:


sc.parallelize([('a',1),('a',2),('c',3),('d',5)]).groupByKey().map(lambda a:(a[0],sum(a[1]))).collect()


# In[92]:


sc.parallelize([('a',1),('a',2),('c',3),('d',5)]).reduceByKey(lambda a,b:a+b).collect()


# ### sortByKey() 返回一个根据键排序的RDD
# ### sortBy() 根据value值进行排序
# ### keys()
# ### values()

# In[97]:


sc.parallelize([('a',1),('a',2),('c',3),('d',5)]).sortByKey().collect()


# In[98]:


sc.parallelize([('a',1),('a',2),('c',3),('d',5)]).sortBy(lambda x:x[0],True).collect()
sc.parallelize([('a',1),('a',2),('c',3),('d',5)]).sortBy(lambda x:x[0],False).collect()


# In[100]:


sc.parallelize([('a',1),('a',2),('c',3),('d',5)]).sortBy(lambda x:x[1],False).collect()


# In[87]:


sc.parallelize([('a',1),('a',2),('c',3),('d',5)]).keys().collect()
sc.parallelize([('a',1),('a',2),('c',3),('d',5)]).values().collect()


# # join() 
# join的类型也和关系数据库中的join一样，包括内连接(join)、左外连接(leftOuterJoin)、右外连接(rightOuterJoin)等。最常用的情形是内连接，所以，join就表示内连接。
# 对于内连接，对于给定的两个输入数据集(K,V1)和(K,V2)，只有在两个数据集中都存在的key才会被输出，最终得到一个(K,(V1,V2))类型的数据集。

# In[101]:


a = sc.parallelize([('a',1),('a',2),('b',3),('b',5)])
b = sc.parallelize([('a','b')])
a.join(b).collect()


# In[110]:


sc.parallelize([('a',1),('a',2),('b',3),('b',5)]).mapValues(lambda x:(x,1)).reduceByKey(lambda a,b:(a[0]+b[0],a[1]+b[1])).mapValues(lambda a:a[0]/a[1]).collect()


# In[134]:


class sort():
    def __init__(self,k):
        self.a=k[0]
        self.b=k[0]
    def __gt__(self,other):
        if other.a==self.a:
            return gt(self.b,other.b)
        else: return gt(self.a,other.a)
sc.parallelize(['5 3','5 1','8 9','8 3','4 7','5 6','3 2']).map(lambda x:(int(x.split(' ')[0]),int(x.split(' ')[1]))).collect()


# # Spark SQL
# ### DataFrame操作
# * df.printSchema()  打印模式信息
# * df.select(df.name,df.age + 1).show()  选择多列
# * df.filter(df.age > 20 ).show() 条件过滤
# * df.groupBy("age").count().show() 分组聚合
# * df.sort(df.age.desc()).show() 排序
# * df.sort(df.age.desc(), df.name.asc()).show() 多列排序
# * df.select(df.name.alias("username"),df.age).show() 对列进行重命名

# In[14]:


from pyspark.sql import SparkSession
spark=SparkSession.builder.getOrCreate()


# In[2]:


df=spark.read.json('file:///home/hadoop/project/spark/person.json')


# In[3]:


df.printSchema()
df.show()


# In[ ]:


df.select(df("name"),df("age")+1).show()
df.select($"name", $"age" + 1).show()
df.filter($"age">=19).show()
df.filter(df("age") > 20 ).show()
df.groupBy($"age">18).count.show()
df.groupBy("age").count().show()
df.sort(df("age").desc).show()
df.sort(df("age").desc, df("name").asc).show()
df.select(df("name").as("username"),df("age")).show()


# In[6]:


df.select('name').write.format('text').save('/hive/person1.txt')
df.rdd.saveAsTextFile('/hive/person2.txt')


# # spark_dfs

# In[16]:


df = spark.read.orc("hdfs:///hive/zxx.db/t").toDF("a","b")
df.createOrReplaceTempView("date")
# spark.sql("show tables").show()
spark.sql("select * from date").show()


# In[56]:


df.where("a =4").show()
df.filter("a =4").show()
df.selectExpr('a as a1','b as b1').show()
df.drop('a').show()
df.sort(df['a'].).show()
df.sort('a').show()


# # 从RDD转换得到DataFrame
# ### 在利用反射机制推断RDD模式时,我们会用到toDF()方法(提前知道数据结构)

# In[154]:


from pyspark.sql.types import Row


# In[239]:


peopleDF=sc.textFile('file:///home/hadoop/project/spark/person.json').map(lambda line:eval(line)).map(lambda x:Row(name=x['name'],age=x['age'])).toDF()


# In[255]:


peopleDF.createOrReplaceTempView("people")
spark.sql("select * from people").rdd.map(lambda p:'name:'+p.name+','+'age:'+str(p.age)).collect()


# # 使用编程方式定义RDD模式

# In[9]:


spark.stop()


# # 通过JDBC连接数据库

# In[9]:


jdbcDF = spark.read.format("jdbc").option("url", "jdbc:mysql://localhost:3306/zxx").option("driver","com.mysql.jdbc.Driver").option("dbtable", "t").option("user", "root").option("password", "hive").load().show()


# In[10]:


jdbcDF = spark.read.format("jdbc").option("url", "jdbc:mysql://localhost:3306/spark").option("driver","com.mysql.jdbc.Driver").option("dbtable", "student").option("user", "root").option("password", "hive").load().show()


# In[266]:


from pyspark.sql.types import Row
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType


# In[267]:


studentRDD = sc.parallelize(["3 Rongcheng M 26","4 Guanhua M 27"]).map(lambda line : line.split(" "))
studentRDD.collect()


# In[268]:


schema = StructType([StructField("name", StringType(), True),StructField("gender", StringType(), True),StructField("age",IntegerType(), True)])
rowRDD = studentRDD.map(lambda p : Row(p[1].strip(), p[2].strip(),int(p[3])))
studentDF = spark.createDataFrame(rowRDD, schema)


# In[270]:


prop = {}
prop['user'] = 'root'
prop['password'] = 'hive'
prop['driver'] = "com.mysql.jdbc.Driver"
studentDF.write.jdbc("jdbc:mysql://localhost:3306/spark",'student','append', prop)


# # hive on spark

# In[10]:


from pyspark import SparkContext, SparkConf, HiveContext
conf=SparkConf()
conf.setAppName("My app")
sc = SparkContext(conf=conf)


# In[13]:


sqlContext = HiveContext(sc)
my_dataframe = sqlContext.sql('Select * from zxx.t')
my_dataframe = sqlContext.sql("insert into zxx.t select 4,'d'")
sqlContext.sql('Select * from zxx.t').show()


# # Spark Streaming程序基本步骤
# 编写Spark Streaming程序的基本步骤是：
# 1.通过创建输入DStream来定义输入源
# 2.通过对DStream应用转换操作和输出操作来定义流计算。
# 3.用streamingContext.start()来开始接收数据和处理流程。
# 4.通过streamingContext.awaitTermination()方法来等待处理结束（手动结束或因为错误而结束）。
# 5.可以通过streamingContext.stop()来手动结束流计算进程。

# In[ ]:


from pyspark import SparkContext
from pyspark.streaming import StreamingContext
# 1表示每隔1秒钟就自动执行一次流计算，这个秒数可以自由设定。
ssc = StreamingContext(sc, 1)


# * 如果是编写一个独立的Spark Streaming程序，而不是在pyspark中运行，则需要通过如下方式创建StreamingContext对象：

# In[ ]:


from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
conf = SparkConf()
conf.setAppName('TestDStream')
conf.setMaster('local[2]')
sc = SparkContext(conf = conf)
ssc = StreamingContext(sc, 20)


# # 输入源
# ### 基本输入源：
# * 文件流
# * 套接字流
# * RDD队列流
# ### 高级数据源
# * Apache Kafka
# * Apache Flume

# ### 文件流

# In[4]:


from operator import add
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
conf = SparkConf()
conf.setAppName('TestDStream')
conf.setMaster('local[2]')
sc = SparkContext(conf = conf)
ssc = StreamingContext(sc, 10)


# In[5]:


lines = ssc.textFileStream('file://///home/hadoop/project/spark/streaming')
words = lines.flatMap(lambda line: line.split(' '))
wordCounts = words.map(lambda x : (x,1)).reduceByKey(add)
wordCounts.pprint()
ssc.start()
# ssc.awaitTermination()


# In[6]:


ssc.stop() 


# ### RDD队列流

# In[ ]:


import time
 
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
 
if __name__ == "__main__":
 
    sc = SparkContext(appName="PythonStreamingQueueStream")
    ssc = StreamingContext(sc, 1)
 
    # Create the queue through which RDDs can be pushed to
    # a QueueInputDStream
    rddQueue = []
    for i in range(5):
        rddQueue += [ssc.sparkContext.parallelize([j for j in range(1, 1001)], 10)]
 
    # Create the QueueInputDStream and use it do some processing
    inputStream = ssc.queueStream(rddQueue)
    mappedStream = inputStream.map(lambda x: (x % 10, 1))
    reducedStream = mappedStream.reduceByKey(lambda a, b: a + b)
    reducedStream.pprint()
 
    ssc.start()
    time.sleep(6)
    ssc.stop(stopSparkContext=True, stopGraceFully=True)


# # Spark MLlib

# In[15]:


from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer


# In[ ]:


# spark = SparkSession.builder.master("local").appName("Word Count").getOrCreate()


# In[13]:


training = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"])


# In[16]:


tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])


# In[17]:


model = pipeline.fit(training)


# In[18]:


test = spark.createDataFrame([
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "spark hadoop spark"),
    (7, "apache hadoop")
], ["id", "text"])


# In[19]:


prediction = model.transform(test)
selected = prediction.select("id", "text", "probability", "prediction")
for row in selected.collect():
    rid, text, prob, prediction = row
    print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction))


# # 特征变换–标签和索引的转化
# 在机器学习处理过程中，为了方便相关算法的实现，经常需要把标签数据（一般是字符串）转化成整数索引，或是在计算结束后将整数索引还原为相应的标签。
# 
# Spark ML包中提供了几个相关的转换器，例如：StringIndexer、IndexToString、OneHotEncoder、VectorIndexer，它们提供了十分方便的特征转换功能，这些转换器类都位于org.apache.spark.ml.feature包下。
# 
# 值得注意的是，用于特征转换的转换器和其他的机器学习算法一样，也属于ML Pipeline模型的一部分，可以用来构成机器学习流水线，以StringIndexer为例，其存储着进行标签数值化过程的相关 超参数，是一个Estimator，对其调用fit(..)方法即可生成相应的模型StringIndexerModel类，很显然，它存储了用于DataFrame进行相关处理的 参数，是一个Transformer（其他转换器也是同一原理）
# 下面对几个常用的转换器依次进行介绍。

# In[ ]:


StringIndexer
​StringIndexer转换器可以把一列类别型的特征（或标签）进行编码，使其数值化，索引的范围从0开始，该过程可以使得相应的特征索引化，使得某些无法接受类别型特征的算法可以使用，并提高诸如决策树等机器学习算法的效率。

索引构建的顺序为标签的频率，优先编码频率较大的标签，所以出现频率最高的标签为0号。
如果输入的是数值型的，我们会把它转化成字符型，然后再对其进行编码。

首先，引入必要的包，并创建一个简单的DataFrame，它只包含一个id列和一个标签列category：


# In[34]:


from pyspark.ml.feature import IndexToString, StringIndexer
 
df = spark.createDataFrame(
    [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
    ["id", "category"])
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
model = indexer.fit(df)
indexed = model.transform(df)
indexed.show()


# # IndexToString
# 与StringIndexer相对应，IndexToString的作用是把标签索引的一列重新映射回原有的字符型标签。
# 
# 其主要使用场景一般都是和StringIndexer配合，先用StringIndexer将标签转化成标签索引，进行模型训练，然后在预测标签的时候再把标签索引转化成原有的字符标签。当然，你也可以另外定义其他的标签。
# 
# 首先，和StringIndexer的实验相同，我们用StringIndexer读取数据集中的“category”列，把字符型标签转化成标签索引，然后输出到“categoryIndex”列上，构建出新的DataFrame。

# In[35]:


converter = IndexToString(inputCol="categoryIndex", outputCol="originalCategory")
converted = converter.transform(indexed)
converted.select("id", "categoryIndex", "originalCategory").show()


# # VectorIndexer
# 之前介绍的StringIndexer是针对单个类别型特征进行转换，倘若所有特征都已经被组织在一个向量中，又想对其中某些单个分量进行处理时，Spark ML提供了VectorIndexer类来解决向量数据集中的类别性特征转换。
# 
# 通过为其提供maxCategories超参数，它可以自动识别哪些特征是类别型的，并且将原始值转换为类别索引。它基于不同特征值的数量来识别哪些特征需要被类别化，那些取值可能性最多不超过maxCategories的特征需要会被认为是类别型的。
# 
# 在下面的例子中，我们读入一个数据集，然后使用VectorIndexer训练出模型，来决定哪些特征需要被作为类别特征，将类别特征转换为索引，这里设置maxCategories为10，即只有种类小10的特征才被认为是类别型特征，否则被认为是连续型特征：

# In[42]:


from pyspark.ml.feature import VectorIndexer
from pyspark.ml.linalg import Vector,Vectors
df=spark.createDataFrame([(Vectors.dense(-1.0,1.0,1.0),),(Vectors.dense(-1.0,1.0,1.0),),(Vectors.dense(-1.0,1.0,1.0),)],['features'])


# In[45]:


indexer = VectorIndexer(inputCol="features", outputCol="indexed", maxCategories=2)
indexerModel = indexer.fit(df)
categoricalFeatures = indexerModel.categoryMaps.keys()


# # OneHotEncoder
# ​独热编码（One-Hot Encoding） 是指把一列类别性特征（或称名词性特征，nominal/categorical features）映射成一系列的二元连续特征的过程，原有的类别性特征有几种可能取值，这一特征就会被映射成几个二元连续特征，每一个特征代表一种取值，若该样本表现出该特征，则取1，否则取0。
# 
# One-Hot编码适合一些期望类别特征为连续特征的算法，比如说逻辑斯蒂回归等。
# 
# 首先创建一个DataFrame，其包含一列类别性特征，需要注意的是，在使用OneHotEncoder进行转换前，DataFrame需要先使用StringIndexer将原始标签数值化：

# In[47]:


from pyspark.ml.feature import OneHotEncoder, StringIndexer
 
df = spark.createDataFrame([
    (0, "a"),
    (1, "b"),
    (2, "c"),
    (3, "a"),
    (4, "a"),
    (5, "c")
], ["id", "category"])
stringIndexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
model = stringIndexer.fit(df)
indexed = model.transform(df)


# In[48]:


encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
encoded = encoder.transform(indexed)
encoded.show()


# # 特征抽取 — TF-IDF

# “词频－逆向文件频率”（TF-IDF）是一种在文本挖掘中广泛使用的特征向量化方法，它可以体现一个文档中词语在语料库中的重要程度。
# 
# ​ 词语由t表示，文档由d表示，语料库由D表示。词频TF(t,d)是词语t在文档d中出现的次数。文件频率DF(t,D)是包含词语的文档的个数。如果我们只使用词频来衡量重要性，很容易过度强调在文档中经常出现，却没有太多实际信息的词语，比如“a”，“the”以及“of”。如果一个词语经常出现在语料库中，意味着它并不能很好的对文档进行区分。TF-IDF就是在数值化文档信息，衡量词语能提供多少信息以区分文档。
# ​ 在Spark ML库中，TF-IDF被分成两部分：TF (+hashing) 和 IDF。
# 
# TF: HashingTF 是一个Transformer，在文本处理中，接收词条的集合然后把这些集合转化成固定长度的特征向量。这个算法在哈希的同时会统计各个词条的词频。
# 
# IDF: IDF是一个Estimator，在一个数据集上应用它的fit（）方法，产生一个IDFModel。 该IDFModel 接收特征向量（由HashingTF产生），然后计算每一个词在文档中出现的频次。IDF会减少那些在语料库中出现频率较高的词的权重。
# 
# ​ Spark.mllib 中实现词频率统计使用特征hash的方式，原始特征通过hash函数，映射到一个索引值。后面只需要统计这些索引值的频率，就可以知道对应词的频率。这种方式避免设计一个全局1对1的词到索引的映射，这个映射在映射大量语料库时需要花费更长的时间。但需要注意，通过hash的方式可能会映射到同一个值的情况，即不同的原始特征通过Hash映射后是同一个值。为了降低这种情况出现的概率，我们只能对特征向量升维。i.e., 提高hash表的桶数，默认特征维度是 2^20 = 1,048,576.

# In[20]:


from pyspark.ml.feature import HashingTF,IDF,Tokenizer
sentenceData = spark.createDataFrame([(0, "I heard about Spark and I love Spark"), (0, "I wish Java could use case classes"), (1, "Logistic regression models are neat")]).toDF("label", "sentence")


# In[21]:


tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(sentenceData)


# In[22]:


wordsData.show()


# In[30]:


hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=2)
featurizedData = hashingTF.transform(wordsData)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)


# In[31]:


rescaledData = idfModel.transform(featurizedData)
rescaledData.select("label", "features").show()


# # 特征抽取–Word2Vec

# In[49]:


from pyspark.ml.feature import Word2Vec
documentDF = spark.createDataFrame([
    ("Hi I heard about Spark".split(" "), ),
    ("I wish Java could use case classes".split(" "), ),
    ("Logistic regression models are neat".split(" "), )
], ["text"])
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
model = word2Vec.fit(documentDF)
result = model.transform(documentDF)
for row in result.collect():
    text, vector = row
    print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))


# # 特征抽取–CountVectorizer
# CountVectorizer旨在通过计数来将一个文档转换为向量。当不存在先验字典时，Countvectorizer作为Estimator提取词汇进行训练，并生成一个CountVectorizerModel用于存储相应的词汇向量空间。该模型产生文档关于词语的稀疏表示，其表示可以传递给其他算法，例如LDA。
# 
# 在CountVectorizerModel的训练过程中，CountVectorizer将根据语料库中的词频排序从高到低进行选择，词汇表的最大含量由vocabsize超参数来指定，超参数minDF，则指定词汇表中的词语至少要在多少个不同文档中出现。
# 我们默认名为spark的SparkSession已经创建。

# In[54]:


from pyspark.ml.feature import CountVectorizer
df = spark.createDataFrame([
    (0, "a b c".split(" ")),
    (1, "a b b c a".split(" "))
], ["id", "words"])
cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=3, minDF=2.0)
model = cv.fit(df)
result = model.transform(df)
# result.show()
result.show(truncate=False)


# # 特征选取–卡方选择器
# 特征选择（Feature Selection）指的是在特征向量中选择出那些“优秀”的特征，组成新的、更“精简”的特征向量的过程。它在高维数据分析中十分常用，可以剔除掉“冗余”和“无关”的特征，提升学习器的性能。
# 
# 特征选择方法和分类方法一样，也主要分为有监督（Supervised）和无监督（Unsupervised）两种，卡方选择则是统计学上常用的一种有监督特征选择方法，它通过对特征和真实标签之间进行卡方检验，来判断该特征和真实标签的关联程度，进而确定是否对其进行选择。
# 
# 和ML库中的大多数学习方法一样，ML中的卡方选择也是以estimator+transformer的形式出现的，其主要由ChiSqSelector和ChiSqSelectorModel两个类来实现。
# 
# 在进行实验前，首先进行环境的设置。引入卡方选择器所需要使用的类：

# In[55]:


from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors


# In[56]:


df = spark.createDataFrame([
    (7, Vectors.dense([0.0, 0.0, 18.0, 1.0]), 1.0,),
    (8, Vectors.dense([0.0, 1.0, 12.0, 0.0]), 0.0,),
    (9, Vectors.dense([1.0, 0.0, 15.0, 0.1]), 0.0,)], ["id", "features", "clicked"])


# In[57]:


selector = ChiSqSelector(numTopFeatures=1, featuresCol="features",
                         outputCol="selectedFeatures", labelCol="clicked")
result = selector.fit(df).transform(df)
result.show()


# # 逻辑斯蒂回归分类器

# In[58]:


from pyspark.sql import Row,functions
from pyspark.ml.linalg import Vector,Vectors
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer,HashingTF, Tokenizer
from pyspark.ml.classification import LogisticRegression,LogisticRegressionModel,BinaryLogisticRegressionSummary, LogisticRegression


# In[59]:


def f(x):
    rel = {}
    rel['features'] = Vectors.dense(float(x[0]),float(x[1]),float(x[2]),float(x[3]))
    rel['label'] = str(x[4])
    return rel
data = spark.sparkContext.textFile("file:///home/hadoop/project/spark/iris.txt").map(lambda line: line.split(',')).map(lambda p: Row(**f(p))).toDF()


# In[60]:


data.show()


# In[68]:


data.createOrReplaceTempView("iris")
df = spark.sql("select * from iris where label != 'Iris-setosa'")
# rel = df.rdd.map(lambda t : str(t[1])+":"+str(t[0])).collect()
# for item in rel:
#     print(item)


# In[77]:


# 3. 构建ML的pipeline
# ​ 分别获取标签列和特征列，进行索引，并进行了重命名。
# labelIndexer = StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(df)
# featureIndexer = VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(df)
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(df)
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures").fit(df)


# In[70]:


trainingData, testData = df.randomSplit([0.7,0.3])


# In[93]:


lr = LogisticRegression().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
lr = LogisticRegression(labelCol="indexedLabel",featuresCol="indexedFeatures",maxIter=10,regParam=0.3,elasticNetParam=0.8)
# print("LogisticRegression parameters:\n" + lr.explainParams())


# In[94]:


labelConverter = IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
lrPipeline =  Pipeline().setStages([labelIndexer, featureIndexer, lr, labelConverter])
lrPipelineModel = lrPipeline.fit(trainingData)
lrPredictions = lrPipelineModel.transform(testData)
preRel = lrPredictions.select("predictedLabel", "label", "features", "probability").collect()
# for item in preRel:
#     print(str(item['label'])+','+str(item['features'])+'-->prob='+str(item['probability'])+',predictedLabel'+str(item['predictedLabel']))


# In[95]:


evaluator = MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction")
lrAccuracy = evaluator.evaluate(lrPredictions)
print("Test Error = " + str(1.0 - lrAccuracy))


# In[91]:


lrModel = lrPipelineModel.stages[2]
print("Coefficients: " + str(lrModel.coefficients)+'\n'
      +"Intercept: "+str(lrModel.intercept)+'\n'
      +"numClasses: "+str(lrModel.numClasses)+'\n'
      +"numFeatures: "+str(lrModel.numFeatures))


# In[92]:


trainingSummary = lrModel.summary
objectiveHistory = trainingSummary.objectiveHistory
for item in objectiveHistory:
    print(item)


# # 决策树分类器

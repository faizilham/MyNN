package weka_test;

import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

public class Main {
	public static void main(String[] args) throws Exception{
		
		//load nominal arff
		FileReader freader = new FileReader("data/weather.nominal.arff");
		Instances data1 = new Instances(freader);
		data1.setClassIndex(data1.numAttributes()-1);
		freader.close();
		System.out.println(data1.toSummaryString());
		
		
		//load numeric csv
		DataSource source = new DataSource("data/weather.numeric.csv");
		Instances data2 = source.getDataSet();
		data2.setClassIndex(data2.numAttributes()-1);
		System.out.println(data2.toSummaryString());
		
		
		System.out.println("Sebelum remove attribute 2");
		System.out.println(data2.toSummaryString());
		
		data2.deleteAttributeAt(1);
		
		System.out.println("Setelah remove attribute 2");
		System.out.println(data2.toSummaryString());
		
		

		System.out.println("Sebelum resample");
		System.out.println(data1.toSummaryString());
		
		String[] opt = new String[2];
		opt[0] = "-Z";  // resample dengan mode persentase
		opt[1] = "50";  // jumlah data hasil resample dalam persen
		
		Resample resample = new Resample();
		resample.setOptions(opt); 
		//resample.setInputFormat(data1);
		resample.setInputFormat(data1);
		Instances newData = Filter.useFilter(data1, resample); 
		
		System.out.println("Setelah resample");
		System.out.println(newData.toSummaryString());
		
		//naive bayes
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(data1);
		for(int i = 0; i < data1.numInstances(); ++i){
			nb.updateClassifier(data1.instance(i));
		}
		System.out.println(nb.toString());
		
		
		// decision tree
		Id3 tree = new Id3();

		tree.buildClassifier(data1);
		System.out.println(tree.toString());
		
		// load test dataset
		freader = new FileReader("data/weather.test.arff");
		Instances testdata = new Instances(freader);
		testdata.setClassIndex(testdata.numAttributes()-1);
		freader.close();
		
		// evaluate
		Evaluation eval1 = new Evaluation(testdata);
		eval1.evaluateModel(tree, testdata);
		System.out.println(eval1.toSummaryString());
		
		Evaluation eval2 = new Evaluation(data1);
		eval2.crossValidateModel(tree, data1, 10, new Random());
		System.out.println(eval2.toSummaryString());
		
		// build 80% train dataset, 20% test dataset
		int ntrain = data1.numInstances() * 80 / 100;
		int ntest = data1.numInstances() - ntrain;
		
		Instances train = new Instances(data1, 0, ntrain);
		Instances test = new Instances(data1, ntrain, ntest);
		
		// train model
		tree.buildClassifier(train);
		
		// evaluate
		Evaluation eval3 = new Evaluation(train);
		eval3.evaluateModel(tree, test);
		System.out.println(eval3.toSummaryString());
		
		SerializationHelper.write("weatherId3.model", tree);
		
		Classifier classifier = (Classifier) weka.core.SerializationHelper.read("weatherId3.model");
		System.out.println(classifier.toString());

		Instance unseen = new Instance(5);
		String[] attr = {"sunny", "cool", "high", "TRUE"};
		
		for (int i = 0; i < 4; ++i)
			unseen.setValue(data1.attribute(i), attr[i]);
		
		unseen.setDataset(data1);
		System.out.println("Unseen data: " + unseen);
		
		int result = (int) tree.classifyInstance(unseen);
		
		System.out.println("Hasil: " + data1.classAttribute().value(result));
		
		System.out.println("Unseen data: " + unseen);
		RandomClassifier rc = new RandomClassifier();
		
		rc.buildClassifier(data1);
		
		result = (int) rc.classifyInstance(unseen);
		System.out.println("Hasil RandomClassifier: " + data1.classAttribute().value(result));
	}
}

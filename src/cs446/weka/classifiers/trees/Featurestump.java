package cs446.weka.classifiers.trees;
import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

/* CS 446:
 * Edit the makeInstance method (and any other necessary fields) to extract the correct set of features 
 */

public class Featurestump {

    static String[] features;
    private static FastVector zeroOne;
    private static FastVector labels;

    static {
    // Define a list of feature template names
    // We provide two feature templates, corresponding to the first and last letters in the first name
	//features = new String[] { "firstName0", "firstName1","firstName2","firstName3","firstName4" ,"lastName0" ,"lastName1", "lastName2","lastName3","lastName4" };
    features= new String[100];
    for(int i=0;i<100;i++){
    	features[i]="label"+i;
    }
    // For each feature template, create a feature name for each letter in the alphabet (a to z)
    // Store these feature names in a temporary list, feat_temp
	int p=0;
	List<String> feat_temp = new ArrayList<String>();
	for (String f : features) {
	     
		feat_temp.add(f );
	   
	    }

        
 
        
        
    // Replace the list of feature template names with the list of feature names in feat_temp
	features = feat_temp.toArray(new String[feat_temp.size()]);

    // Store binary feature values
	zeroOne = new FastVector(2);
	zeroOne.addElement("1");
	zeroOne.addElement("0");

    // Store class labels
	labels = new FastVector(2);
	labels.addElement("+");
	labels.addElement("-");
    }

   /* public static Instances readData(String fileName) throws Exception {

	Instances instances = initializeAttributes();
	Scanner scanner = new Scanner(new File(fileName));

	while (scanner.hasNextLine()) {
	    String line = scanner.nextLine();

	    Instance instance = makeInstance(instances, line);

	    instances.add(instance);
	}

	scanner.close();

	return instances;
    }
  */
    private static Instances initializeAttributes() {

	String nameOfDataset = "Badges";

	Instances instances;

	FastVector attributes = new FastVector(features.length+1);
	for (String featureName : features) {
	    attributes.addElement(new Attribute(featureName, zeroOne));
	}
	Attribute classLabel = new Attribute("Class", labels);
	attributes.addElement(classLabel);

	instances = new Instances(nameOfDataset, attributes, 0);

	instances.setClass(classLabel);

	return instances;

    }
    
    
    /**
     * Construct an Instance from an input line from a data file
     * instances: the set of Weka instances we'd like to add a new Instance to
     * inputLine: line in data file, for example "+ naoki abe", which we need to convert to an Instance object
     */
    private static Instance makeInstance(Instances instances, String inputLine) {
	/*inputLine = inputLine.trim();

    // Parse inputLine -- you will need to store the last name, too
	String[] parts = inputLine.split("\\s+");
	String label = parts[0];
	String firstName = parts[1].toLowerCase();
        
        
    String lastName=parts[2].toLowerCase();*/
         
        
        

    // Instantiate the Instance object, and add it to instances
	Instance instance = new Instance(features.length + 1);
	instance.setDataset(instances);

    // Store the set of active features for this name    
	Set<String> feats = new HashSet<String>();
        for(int j=0;j<=99;j++){ 
	feats.add("label"+j);
           
            
        }
       /* for(int j=0;j<=lastName.length()-1;j++){
            
        	feats.add("lastName"+j+"=" + lastName.charAt(j++));
                   
                    
                }*/

  
    // Set the value of each feature for this example
	for (int featureId = 0; featureId < features.length; featureId++) {
	    Attribute att = instances.attribute(features[featureId]);
	    String name = att.name();
        // If feature is active, value is "1"; if inactive, value is "0"
	    String featureLabel;
	    if (feats.contains(name)) {
		featureLabel = "1";
	    } else
		featureLabel = "0";
	    instance.setValue(att, featureLabel);
	}
    // Set the example's class label
	instance.setClassValue(label);

    // Return the new instance
	return instance;
    }
   
   

    public static void main(String[] args) throws Exception {

	if (args.length != 2) {
	    System.err
		    .println("Usage: FeatureGenerator input-badges-file features-file");
	    System.exit(-1);
	}
	Instances data = readData(args[0]);

	ArffSaver saver = new ArffSaver();
	saver.setInstances(data);
	saver.setFile(new File(args[1]));
	saver.writeBatch();
    }
}

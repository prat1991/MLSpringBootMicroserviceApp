package prat.classifier.model;

public class OutputTargetResponse {
    private String species;
    private double confidence;
    
    // Default constructor
    public OutputTargetResponse() {}
    
    // Constructor with parameters
    public OutputTargetResponse(String species, double confidence) {
        this.species = species;
        this.confidence = confidence;
    }
    
    // Getters and setters
    public String getSpecies() {
        return species;
    }
    
    public void setSpecies(String species) {
        this.species = species;
    }
    
    public double getConfidence() {
        return confidence;
    }
    
    public void setConfidence(double confidence) {
        this.confidence = confidence;
    }
}
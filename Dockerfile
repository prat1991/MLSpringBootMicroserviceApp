# Use the official OpenJDK image as a parent image
FROM openjdk:17-jdk-slim

# Set working directory
WORKDIR /app

# Copy the jar file into the image
COPY target/*.jar app.jar

# Run the jar file
ENTRYPOINT ["java","-jar","/app/app.jar"]

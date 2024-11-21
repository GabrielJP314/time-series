# Loading the us_change dataset

# Install the necessary packages
install.packages("fpp3")

# Load the packages
library(fpp3)

# Save the data to csv
write.csv(us_change, "us_change.csv", row.names = FALSE)

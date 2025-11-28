#include "csv.h"

#include <sstream>
#include <fstream>
#include <cmath>
#include <algorithm>

#include <iostream>

CSV::CSV(const char * filename, int a) {
	if (a == 0)
		importHurr(filename);
	else if (a==1)
		importDriver(filename);
	else
		importDriverMany(filename, a);
}

CSV::~CSV() {

}

// Process a config file line
std::vector<std::string>& CSV::split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

// Process a config file line
std::vector<std::string> CSV::split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

void CSV::importDriverMany(const char * filename, int size) {
	std::ifstream infile(filename);

	// this doesn't have checking for format errors/values
	// skip first line

	
	for (std::string line; getline(infile,line);) {
		std::vector<std::string> x = split(line, ',');
		driver.push_back(::atof(x[0].c_str()));

	}
	
}
void CSV::importDriver(const char * filename) {
	std::ifstream infile(filename);

	// this doesn't have checking for format errors/values
	// skip first line

	
	for (std::string line; getline(infile,line);) {
		std::vector<std::string> x = split(line, ',');
		driver.push_back(::atof(x[1].c_str()));

	}
	
}

void CSV::importHurr(const char * filename) {
	std::ifstream infile(filename);

	// this doesn't have checking for format errors/values
	// skip first line

	unsigned int i = 0;
	for (std::string line; getline(infile,line);) {
		if (i<1) {
			i++;
			continue;
		}

		std::vector<std::string> x = split(line, ',');
		year.push_back(std::stoul(x[1]));
		month.push_back(std::stoul(x[2]));
		day.push_back(std::stoul(x[3]));
		hour.push_back(std::stoul(x[4]));
		lat.push_back(::atof(x[5].c_str()));
		lon.push_back(::atof(x[6].c_str()));
		wind.push_back(std::stoul(x[7]));
		pressure.push_back(std::stoul(x[8]));
		count.push_back(std::stoul(x[10]));

		i++;
	}
}

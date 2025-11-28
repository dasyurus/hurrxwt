#include <string>
#include <vector>

class CSV
{
public:
	CSV(const char * filename, int a);
	~CSV();

	// vectors with useful variables
	std::vector<int> year, month, day, hour, wind, pressure, count;
	std::vector<float> lat, lon, driver;

private:
	void importHurr(const char * filename);
	void importDriver(const char * filename);
	void importDriverMany(const char * filename, int size);
	std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);
	std::vector<std::string> split(const std::string &s, char delim);
};

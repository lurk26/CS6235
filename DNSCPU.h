#pragma once

#include <vector>

class DNSCPU
{
public:
	DNSCPU();
	~DNSCPU();

	enum SIMSTATES
	{
		INIT,
		STEP,
		FINISH,
		END
	};

	static void RunSimulation();

	static std::vector<float>& getPressure();
	static float getPressureWidth();

	static std::vector<float>& getTemperature();
	static float getTemperatureWidth();

private:

	// Call this to initialize shit
	static void Init();

	// Call this to move ahead in time. dt is defined in the .cpp file
	static void DNSCPU::Step();
	
	// Call this to Finish 
	static void DNSCPU::Finish();



};


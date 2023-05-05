

#include "Main.h"


int main(void)
{
	if (Get_FileAddrList() == false)
	{
		cout << "Get_FileAddrList Fail" << endl;

		return -1;
	}


	for (int i = 0; i < vFileAddrList.size(); i++)
	{
		cout << vFileAddrList[i] << endl;
	}


	return 0;
}


bool Get_FileAddrList()
{
	FILE* fp = nullptr;

	char buffer[100];
	string fileAddr;


	fopen_s(&fp, DATA_FILE_ADDR, "r" );
	if (fp == nullptr)
	{
		return false;
	}

	
	while (fscanf_s( fp, "%s\n", buffer, (unsigned int)sizeof(buffer) ) != EOF)
	{
		fileAddr = buffer;
		vFileAddrList.emplace_back(fileAddr);
	}


	if (fp == nullptr)
	{
		return false;
	}

	fclose(fp);
	fp = nullptr;


	if (vFileAddrList.size() <= 0)
	{
		return false;
	}


	return true;
}
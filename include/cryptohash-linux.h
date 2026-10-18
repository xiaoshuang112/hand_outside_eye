#define __linux__ 1

#ifdef _WIN32
// Windows
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <winsock2.h>
#include <iphlpapi.h>
#include <tchar.h>
#include <intrin.h>
#include "time.h"
#include <corecrt.h>
#include <corecrt_wtime.h>

#pragma comment(lib,"ws2_32.lib")
#pragma comment(lib, "IPHLPAPI.lib")

#elif __ANDROID__
// android
#include <jni.h>
#include <string>

#elif __linux__
// linux
#include <cstdio>
#include <string.h>
#include "time.h"
#endif


#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <fstream>


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>
#include <array>
#include <memory>

typedef struct {
	int     tm_sec;         /* seconds */
	int     tm_min;         /* minutes */
	int     tm_hour;        /* hours */
	int     tm_mday;        /* day of the month */
	int     tm_mon;         /* month */
	int     tm_year;        /* year */
	int     tm_wday;        /* day of the week */
	int     tm_yday;        /* day in the year */
	int     tm_isdst;       /* daylight saving time */

	long int tm_gmtoff;     /* Seconds east of UTC.  */
	const char *tm_zone;    /* Timezone abbreviation.  */

}tm_tp;

using namespace std;

//核心类
//人脸检测 + 手检测 + 换脸
#define CORE_FACE 0x01
//性别年龄
#define CORE_AGE 0x02
//人脸识别
#define CORE_FR 0x04
//姿态检测 + 人
#define CORE_POSE 0x08
//OCR
#define CORE_OCR 0x10

//ALL 128
#define CORE_ALL 0x80



//应用类
//大屏
#define APP_DP 1
#define APP_PB 2
#define APP_RED 3
#define APP_WH 4

#define ADD_VECTOR_END(v,i) (v).push_back((i))

#define MALLOC(x) HeapAlloc(GetProcessHeap(), 0, (x))
#define FREE(x) HeapFree(GetProcessHeap(), 0, (x))
#define INVALID 0

static int g_privilege = 0;
static std::vector<std::string> g_list_dev;
static std::vector<std::string> g_list_dev_md5;
static std::vector<std::string> g_list_mac;

//std::string mac_list[] = {
//  "0C9D920E511B", //titan
//  "38D5477C8B93", //sever
//  "F4939FEAB92B", //教室
//  "2CFDA1E01380", //i9
//  "E0D55E7F2BCC", //大屏
//  "60EE5C359197", //大屏
//  "1C3947C34DD5", //xiaoge
//  "4CCC6A6BACD3", //凯丽
//  "4CCC6A2F331C"  //980TI
//
//};

// #define __LOG


static unsigned char  HEADER_KEY[32] = { 0xf1,0x8a,0x75,0xf5, 0xce,0xfd, 0x49,0xe4, 0x89, 0xfb, 0x2b, 0xd3, 0x7e, 0x18, 0x40, 0x9a,
0x95, 0xbc, 0x82, 0xfb, 0xeb, 0x69, 0x40, 0xa0, 0x9b, 0xd4, 0x8e, 0x8b, 0x94, 0x83, 0x26, 0x8d };

const unsigned char HEADER_LEN = 32;

static unsigned char  HEADERBYTE[] =
{ 0xf6, 0xaf, 0x4c, 0xfc, 0x2a, 0x45, 0x46, 0x19, 0x93, 0x5b, 0xba, 0x66, 0x35, 0xdf, 0xf2, 0x42,
	0x15, 0x86, 0x38, 0x2a, 0xd2, 0xff, 0x44, 0x49, 0x98, 0x18, 0x9c, 0xa3, 0x97, 0xd3, 0xde, 0x21 };

static unsigned char  HEADERMERGE[] = { 0xef, 0xc9, 0x4b, 0xdf, 0x2e, 0x6b, 0x45, 0x07, 0x87, 0x5d, 0xae, 0xf7, 0xce, 0x4c, 0xce, 0xed,
0x77, 0xfd, 0xf9, 0xc1, 0x24, 0xa6, 0x40, 0xc4, 0x81, 0x9b, 0x3e, 0x6, 0x2c, 0x48, 0x1, 0xf9 };




static std::string g_devid = "";

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" 
{
#endif

	namespace alva_str_util
	{
		inline string changeCase(string &value, bool lowerCase) {
			int32_t len = value.length();
			string newvalue(value);
			for (string::size_type i = 0, l = newvalue.length(); i < l; ++i)
				newvalue[i] = lowerCase ? tolower(newvalue[i]) : toupper(newvalue[i]);
			return newvalue;
		}

		inline void replace(string &target, string search, string replacement) {
			if (search == replacement)
				return;
			if (search == "")
				return;
			string::size_type i = string::npos;
			string::size_type lastPos = 0;
			while ((i = target.find(search, lastPos)) != string::npos) {
				target.replace(i, search.length(), replacement);
				lastPos = i + replacement.length();
			}
		}

		inline string lowerCase(string value) {
			return changeCase(value, true);
		}

		inline string upperCase(string value) {
			return changeCase(value, false);
		}

		inline void lTrim(string &value) {
			string::size_type i = 0;
			for (i = 0; i < value.length(); i++) {
				if (value[i] != ' ' &&
					value[i] != '\t' &&
					value[i] != '\n' &&
					value[i] != '\r')
					break;
			}
			value = value.substr(i);
		}

		inline void rTrim(string &value) {
			int32_t i = 0;
			for (i = (int32_t)value.length() - 1; i >= 0; i--) {
				if (value[i] != ' ' &&
					value[i] != '\t' &&
					value[i] != '\n' &&
					value[i] != '\r')
					break;
			}
			value = value.substr(0, i + 1);
		}

		inline void trim(string &value) {
			lTrim(value);
			rTrim(value);
		}


		inline void split(string str, string separator, vector<string> &result) {
			result.clear();
			string::size_type position = str.find(separator);
			string::size_type lastPosition = 0;
			uint32_t separatorLength = separator.length();

			while (position != str.npos) {
				ADD_VECTOR_END(result, str.substr(lastPosition, position - lastPosition));
				lastPosition = position + separatorLength;
				position = str.find(separator, lastPosition);
			}
			ADD_VECTOR_END(result, str.substr(lastPosition, string::npos));
		}

		inline long long  hextoint64(std::string& hex)
		{
			/*
			long long x;
			std::stringstream ss;
			//ss << std::hex << hex;
			ss << std::hex << "40e20100";
			ss >> x;

			return static_cast<long long>(x);
*/

			std::string s = "0x" + hex;
			long long  x = std::stoll(s, nullptr, 16);

			return x;
		}
		inline std::string int64tohexstr(long long x)
		{
			std::stringstream stream;
			stream << std::hex << x;
			std::string result(stream.str());

			return result;
		}


		inline std::string hextostr(char* hexval, int hex_len, bool uppercase = false)
		{
			std::stringstream sstr;

			if (NULL!=hexval)
			{
				sstr.setf(std::ios_base::hex, std::ios::basefield);
				if (uppercase)
					sstr.setf(std::ios_base::uppercase);
				sstr.fill('0');
				for (size_t i = 0; i < hex_len; i++)
				{
					sstr << std::setw(2) << (unsigned int)(unsigned char)(hexval[i]);
				}
			}

			return sstr.str();
		}
	}
#ifdef _WIN32
	// Windows
	inline unsigned long ip_to_int(const char * ip_src)
	{
		char strIP[16] = { 0 };
		memset(strIP, 0, 16);
		strncpy_s(strIP, ip_src, sizeof(IP_ADDRESS_STRING));
		strIP[15] = 0;

		/* The return value. */
		unsigned long v = 0;
		/* The count of the number of bytes processed. */
		int i;
		/* A pointer to the next digit to process. */
		const char * start;

		start = strIP;
		for (i = 0; i < 4; i++) {
			/* The digit being processed. */
			char c;
			/* The value of this byte. */
			int n = 0;
			while ((start - strIP) < 15) {
				c = *start;
				start++;
				if (c >= '0' && c <= '9') {
					n *= 10;
					n += c - '0';
				}
				/* We insist on stopping at "." if we are still parsing
				   the first, second, or third numbers. If we have reached
				   the end of the numbers, we will allow any character. */
				else if ((i < 3 && c == '.') || i == 3) {
					break;
				}
				else {
					return INVALID;
				}
			}
			if (n >= 256) {
				return INVALID;
			}
			v *= 256;
			v += n;
		}
		return v;

	}

	inline int GetCPUID(std::string &id_buf)
	{
		id_buf = "";
		int cpuinfo[4] = { 0, 0, 0, 0 };
		__cpuid(cpuinfo, 0);

		char buf[1024] = { 0 };

		sprintf_s(buf, "%d%d%d%d", cpuinfo[0], cpuinfo[1], cpuinfo[2], cpuinfo[3]);

		id_buf = buf;
		return 0;
	}
	inline int GetDiskID(std::string &id_buf)
	{
		id_buf = "";
		unsigned long  serialNum = 0;

		// Determine if this volume uses an NTFS file system.      
		GetVolumeInformationA("c:\\", NULL, 0, &serialNum, NULL, NULL, NULL, 0);

		char buf[1024] = { 0 };

		sprintf_s(buf, "%lu", serialNum);

		id_buf = buf;
		return 0;
	}

	inline int GetVideoID(std::string &id_buf)
	{
		DISPLAY_DEVICEA dd;
		dd.cb = sizeof(DISPLAY_DEVICEA);

		int i = 0;
		id_buf = "";

		// locate primary display device
		while (EnumDisplayDevicesA(NULL, i, &dd, 0))
		{
			if (dd.StateFlags & DISPLAY_DEVICE_PRIMARY_DEVICE)
			{
				id_buf = dd.DeviceID;
				break;
			}

			i++;
		}

		if (id_buf == "") return -1;

		return 0;
	}


	inline int GetIP_MAC(char *ip_buf, char *mac_buf = NULL, char *gateway = NULL, char *wlan_buf = NULL)
	{
		/* Declare and initialize variables */

		// It is possible for an adapter to have multiple
		// IPv4 addresses, gateways, and secondary WINS servers
		// assigned to the adapter. 
		//
		// Note that this sample code only prints out the 
		// first entry for the IP address/mask, and gateway, and
		// the primary and secondary WINS server for each adapter. 

		PIP_ADAPTER_INFO pAdapterInfo;
		PIP_ADAPTER_INFO pAdapter = NULL;
		DWORD dwRetVal = 0;
		UINT i;
		ULONG total = 0;

		if (NULL == ip_buf)
		{
			return 0;
		}

		ip_buf[0] = 0;

		if (NULL != mac_buf)
		{
			mac_buf[0] = 0;
		}


		if (NULL != wlan_buf)
		{
			wlan_buf[0] = 0;
		}

		if (NULL != gateway)
		{
			gateway[0] = 0;
		}

		ULONG ulOutBufLen = sizeof(IP_ADAPTER_INFO);
		pAdapterInfo = (IP_ADAPTER_INFO *)MALLOC(sizeof(IP_ADAPTER_INFO));
		if (pAdapterInfo == NULL) {
			//printf("Error allocating memory needed to call GetAdaptersinfo\n");
			return -1;
		}
		// Make an initial call to GetAdaptersInfo to get
		// the necessary size into the ulOutBufLen variable
		if (GetAdaptersInfo(pAdapterInfo, &ulOutBufLen) == ERROR_BUFFER_OVERFLOW) {
			FREE(pAdapterInfo);
			pAdapterInfo = (IP_ADAPTER_INFO *)MALLOC(ulOutBufLen);
			if (pAdapterInfo == NULL) {
				//printf("Error allocating memory needed to call GetAdaptersinfo\n");
				return -1;
			}
		}

		if ((dwRetVal = GetAdaptersInfo(pAdapterInfo, &ulOutBufLen)) == NO_ERROR) {

			pAdapter = pAdapterInfo;
			while (pAdapter) {

				// 无线网卡
				if (IF_TYPE_IEEE80211 == pAdapter->Type)
				{
					if (NULL != wlan_buf && 0 == wlan_buf[0])
					{
						// validate ip
						total = ip_to_int(pAdapter->IpAddressList.IpAddress.String);
						if (total != INVALID)
						{
							strcpy_s(wlan_buf, 16, pAdapter->IpAddressList.IpAddress.String);
						}
					}
				}
				else if (MIB_IF_TYPE_ETHERNET == pAdapter->Type)
				{
					// 有线网卡
					// validate ip
					total = ip_to_int(pAdapter->IpAddressList.IpAddress.String);
					if (total != INVALID)
					{
						strcpy_s(ip_buf, 16, pAdapter->IpAddressList.IpAddress.String);

						if (NULL != gateway)
						{
							strcpy_s(gateway, 16, pAdapter->GatewayList.IpAddress.String);
						}

						if (NULL != mac_buf)
						{
							char *pMac = mac_buf;

							for (UINT i = 0; i < pAdapter->AddressLength; i++)
							{
								_snprintf_s(pMac, sizeof(pMac), 3, "%02X:", pAdapter->Address[i]);

								pMac += 3;
							}

							pMac--;
							*pMac = 0;
						}

					}
					else
					{
						if (NULL != mac_buf && 0 == mac_buf[0])
						{
							char *pMac = mac_buf;

							for (UINT i = 0; i < pAdapter->AddressLength; i++)
							{
								_snprintf_s(pMac, sizeof(pMac), 3, "%02X:", pAdapter->Address[i]);

								pMac += 3;
							}

							pMac--;
							*pMac = 0;
						}
					}
				}

				// 如果已经获取了 有线和无线网卡的信息就退出
				if (NULL != ip_buf && 0 != ip_buf[0] && (NULL == wlan_buf || (NULL != wlan_buf && 0 != wlan_buf[0])))
				{
					break;
				}

				// 下一个网卡信息
				pAdapter = pAdapter->Next;

			}

		}
		else {
			//printf("GetAdaptersInfo failed with error: %d\n", dwRetVal);

		}

		if (pAdapterInfo)
			FREE(pAdapterInfo);

		return 0;
	}

#elif __ANDROID__
	// android
	inline int GetIP_MAC(char *ip_buf, char *mac_buf = NULL, char *gateway = NULL, char *wlan_buf = NULL) {

	    strcpy(mac_buf,g_devid.c_str());
		return 0;
	}
#elif __linux__
	// linux
inline int GetCPUID(std::string &id_buf)
{
	id_buf = "";

	return 0;
}
inline int GetDiskID(std::string &id_buf)
{
	id_buf = "";
	return 0;

	char pc_info[512] = { 0 };

	return 0;
}

inline int GetVideoID(std::string &id_buf)
{
	id_buf = "";
	return 0;
}


 
inline std::vector<std::string> execute_command(const std::string& cmd) {
    std::vector<std::string> output;
    std::array<char, 128> buffer;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        output.emplace_back(buffer.data());
    }
    return output;
}
 
inline std::vector<std::string> get_mac_address_list() {
    std::string cmd = "ifconfig ";
    auto output = execute_command(cmd);

    std::vector<std::string> list_mac;
    for (const auto& line : output) {
        if (line.find("ether") != std::string::npos) {
            std::istringstream iss(line);
            std::string token;
            while (iss >> token) {
                if (token == "ether") {
                    iss >> token; // MAC address
                    list_mac.emplace_back(token);
                    break;
                }
            }
        }
    }

    return list_mac;
}


inline int GetIP_MAC(char *ip_buf, char *mac_buf = NULL, char *gateway = NULL, char *wlan_buf = NULL)
{
	if (NULL == mac_buf)return-1;

	char pc_info[512] = { 0 };

	return 0;
}

#endif


	inline int GetDevID(std::string &dev_id)
	{
		char ip_buf[20] = { 0 };
		char mac_buf[40] = { 0 };
		std::string cpu_id = "";
		std::string video_id = "";
		std::string disk_id = "";
		std::string mac = "";

		int ret = GetIP_MAC(ip_buf, mac_buf);

		mac = mac_buf;

		dev_id = mac + cpu_id + disk_id + video_id;

		return ret;
	}


	//int ValidatMAC()
	//{
	//	char ip_buf[20] = { 0 };
	//	char mac_buf[40] = { 0 };

	//	GetIP_MAC(ip_buf, mac_buf);

	//	std::string dst_mac = mac_buf;

	//	alva_str_util::replace(dst_mac, ":", "");
	//	alva_str_util::replace(dst_mac, "-", "");
	//	dst_mac = alva_str_util::upperCase(dst_mac);

	//	int str_len = dst_mac.length();

	//	for each (std::string mac in mac_list)
	//	{
	//		if (str_len != mac.length()) continue;

	//		size_t i = 0;
	//		for (i = 0; i < str_len; i++)
	//		{
	//			if (mac[i] != dst_mac[i]) break;
	//		}

	//		if (i == str_len) return 1;
	//	}

	//	return -1;
	//}

	typedef struct
	{
		unsigned int count[2];
		unsigned int state[4];
		unsigned char buffer[64];
	}MD5_CTX;

	static unsigned char PADDING[] = { 0x80,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };

#define F(x,y,z) ((x & y) | (~x & z))
#define G(x,y,z) ((x & z) | (y & ~z))
#define H(x,y,z) (x^y^z)
#define I(x,y,z) (y ^ (x | ~z))
#define ROTATE_LEFT(x,n) ((x << n) | (x >> (32-n)))
#define FF(a,b,c,d,x,s,ac) \
          { \
          a += F(b,c,d) + x + ac; \
          a = ROTATE_LEFT(a,s); \
          a += b; \
          }
#define GG(a,b,c,d,x,s,ac) \
          { \
          a += G(b,c,d) + x + ac; \
          a = ROTATE_LEFT(a,s); \
          a += b; \
          }
#define HH(a,b,c,d,x,s,ac) \
          { \
          a += H(b,c,d) + x + ac; \
          a = ROTATE_LEFT(a,s); \
          a += b; \
          }
#define II(a,b,c,d,x,s,ac) \
          { \
          a += I(b,c,d) + x + ac; \
          a = ROTATE_LEFT(a,s); \
          a += b; \
          }                




	inline void MD5Init_local(MD5_CTX *context)
	{
		context->count[0] = 0;
		context->count[1] = 0;
		context->state[0] = 0x67452301;
		context->state[1] = 0xEFCDAB89;
		context->state[2] = 0x98BADCFE;
		context->state[3] = 0x10325476;
	}

	inline void MD5Encode_local(unsigned char *output, unsigned int *input, unsigned int len)
	{
		unsigned int i = 0, j = 0;
		while (j < len)
		{
			output[j] = input[i] & 0xFF;
			output[j + 1] = (input[i] >> 8) & 0xFF;
			output[j + 2] = (input[i] >> 16) & 0xFF;
			output[j + 3] = (input[i] >> 24) & 0xFF;
			i++;
			j += 4;
		}
	}
	inline void MD5Decode_local(unsigned int *output, unsigned char *input, unsigned int len)
	{
		unsigned int i = 0, j = 0;
		while (j < len)
		{
			output[i] = (input[j]) |
				(input[j + 1] << 8) |
				(input[j + 2] << 16) |
				(input[j + 3] << 24);
			i++;
			j += 4;
		}
	}
	inline void MD5Transform_local(unsigned int state[4], unsigned char block[64])
	{
		unsigned int a = state[0];
		unsigned int b = state[1];
		unsigned int c = state[2];
		unsigned int d = state[3];
		unsigned int x[64];
		MD5Decode_local(x, block, 64);
		FF(a, b, c, d, x[0], 7, 0xd76aa478); /* 1 */
		FF(d, a, b, c, x[1], 12, 0xe8c7b756); /* 2 */
		FF(c, d, a, b, x[2], 17, 0x242070db); /* 3 */
		FF(b, c, d, a, x[3], 22, 0xc1bdceee); /* 4 */
		FF(a, b, c, d, x[4], 7, 0xf57c0faf); /* 5 */
		FF(d, a, b, c, x[5], 12, 0x4787c62a); /* 6 */
		FF(c, d, a, b, x[6], 17, 0xa8304613); /* 7 */
		FF(b, c, d, a, x[7], 22, 0xfd469501); /* 8 */
		FF(a, b, c, d, x[8], 7, 0x698098d8); /* 9 */
		FF(d, a, b, c, x[9], 12, 0x8b44f7af); /* 10 */
		FF(c, d, a, b, x[10], 17, 0xffff5bb1); /* 11 */
		FF(b, c, d, a, x[11], 22, 0x895cd7be); /* 12 */
		FF(a, b, c, d, x[12], 7, 0x6b901122); /* 13 */
		FF(d, a, b, c, x[13], 12, 0xfd987193); /* 14 */
		FF(c, d, a, b, x[14], 17, 0xa679438e); /* 15 */
		FF(b, c, d, a, x[15], 22, 0x49b40821); /* 16 */

											   /* Round 2 */
		GG(a, b, c, d, x[1], 5, 0xf61e2562); /* 17 */
		GG(d, a, b, c, x[6], 9, 0xc040b340); /* 18 */
		GG(c, d, a, b, x[11], 14, 0x265e5a51); /* 19 */
		GG(b, c, d, a, x[0], 20, 0xe9b6c7aa); /* 20 */
		GG(a, b, c, d, x[5], 5, 0xd62f105d); /* 21 */
		GG(d, a, b, c, x[10], 9, 0x2441453); /* 22 */
		GG(c, d, a, b, x[15], 14, 0xd8a1e681); /* 23 */
		GG(b, c, d, a, x[4], 20, 0xe7d3fbc8); /* 24 */
		GG(a, b, c, d, x[9], 5, 0x21e1cde6); /* 25 */
		GG(d, a, b, c, x[14], 9, 0xc33707d6); /* 26 */
		GG(c, d, a, b, x[3], 14, 0xf4d50d87); /* 27 */
		GG(b, c, d, a, x[8], 20, 0x455a14ed); /* 28 */
		GG(a, b, c, d, x[13], 5, 0xa9e3e905); /* 29 */
		GG(d, a, b, c, x[2], 9, 0xfcefa3f8); /* 30 */
		GG(c, d, a, b, x[7], 14, 0x676f02d9); /* 31 */
		GG(b, c, d, a, x[12], 20, 0x8d2a4c8a); /* 32 */

											   /* Round 3 */
		HH(a, b, c, d, x[5], 4, 0xfffa3942); /* 33 */
		HH(d, a, b, c, x[8], 11, 0x8771f681); /* 34 */
		HH(c, d, a, b, x[11], 16, 0x6d9d6122); /* 35 */
		HH(b, c, d, a, x[14], 23, 0xfde5380c); /* 36 */
		HH(a, b, c, d, x[1], 4, 0xa4beea44); /* 37 */
		HH(d, a, b, c, x[4], 11, 0x4bdecfa9); /* 38 */
		HH(c, d, a, b, x[7], 16, 0xf6bb4b60); /* 39 */
		HH(b, c, d, a, x[10], 23, 0xbebfbc70); /* 40 */
		HH(a, b, c, d, x[13], 4, 0x289b7ec6); /* 41 */
		HH(d, a, b, c, x[0], 11, 0xeaa127fa); /* 42 */
		HH(c, d, a, b, x[3], 16, 0xd4ef3085); /* 43 */
		HH(b, c, d, a, x[6], 23, 0x4881d05); /* 44 */
		HH(a, b, c, d, x[9], 4, 0xd9d4d039); /* 45 */
		HH(d, a, b, c, x[12], 11, 0xe6db99e5); /* 46 */
		HH(c, d, a, b, x[15], 16, 0x1fa27cf8); /* 47 */
		HH(b, c, d, a, x[2], 23, 0xc4ac5665); /* 48 */

											  /* Round 4 */
		II(a, b, c, d, x[0], 6, 0xf4292244); /* 49 */
		II(d, a, b, c, x[7], 10, 0x432aff97); /* 50 */
		II(c, d, a, b, x[14], 15, 0xab9423a7); /* 51 */
		II(b, c, d, a, x[5], 21, 0xfc93a039); /* 52 */
		II(a, b, c, d, x[12], 6, 0x655b59c3); /* 53 */
		II(d, a, b, c, x[3], 10, 0x8f0ccc92); /* 54 */
		II(c, d, a, b, x[10], 15, 0xffeff47d); /* 55 */
		II(b, c, d, a, x[1], 21, 0x85845dd1); /* 56 */
		II(a, b, c, d, x[8], 6, 0x6fa87e4f); /* 57 */
		II(d, a, b, c, x[15], 10, 0xfe2ce6e0); /* 58 */
		II(c, d, a, b, x[6], 15, 0xa3014314); /* 59 */
		II(b, c, d, a, x[13], 21, 0x4e0811a1); /* 60 */
		II(a, b, c, d, x[4], 6, 0xf7537e82); /* 61 */
		II(d, a, b, c, x[11], 10, 0xbd3af235); /* 62 */
		II(c, d, a, b, x[2], 15, 0x2ad7d2bb); /* 63 */
		II(b, c, d, a, x[9], 21, 0xeb86d391); /* 64 */
		state[0] += a;
		state[1] += b;
		state[2] += c;
		state[3] += d;
	}

	inline void MD5Update_local(MD5_CTX *context, unsigned char *input, unsigned int inputlen)
	{
		unsigned int i = 0, index = 0, partlen = 0;
		index = (context->count[0] >> 3) & 0x3F;
		partlen = 64 - index;
		context->count[0] += inputlen << 3;
		if (context->count[0] < (inputlen << 3))
			context->count[1]++;
		context->count[1] += inputlen >> 29;

		if (inputlen >= partlen)
		{
			memcpy(&context->buffer[index], input, partlen);
			MD5Transform_local(context->state, context->buffer);
			for (i = partlen; i + 64 <= inputlen; i += 64)
				MD5Transform_local(context->state, &input[i]);
			index = 0;
		}
		else
		{
			i = 0;
		}
		memcpy(&context->buffer[index], &input[i], inputlen - i);
	}
	inline void MD5Final_local(MD5_CTX *context, unsigned char digest[16])
	{
		unsigned int index = 0, padlen = 0;
		unsigned char bits[8];
		index = (context->count[0] >> 3) & 0x3F;
		padlen = (index < 56) ? (56 - index) : (120 - index);
		MD5Encode_local(bits, context->count, 8);
		MD5Update_local(context, PADDING, padlen);
		MD5Update_local(context, bits, 8);
		MD5Encode_local(digest, context->state, 16);
	}


	static char g_md5[33] = { 0 };
	inline char* GetMD5_no_encode(char* src_str)
	{
		int str_len = strlen(src_str);
		char enc_str[16] = { 0 };
		MD5_CTX md5;
		MD5Init_local(&md5);
		MD5Update_local(&md5, (unsigned char*)src_str, str_len);
		MD5Final_local(&md5, (unsigned char*)enc_str);

		std::string dst_md5 = alva_str_util::hextostr(enc_str, 16);

		for (int i = 0; i < 32; i++)
		{
			g_md5[i] = dst_md5[i];
		}
		g_md5[32] = 0;

		return g_md5;
	}
	inline char* GetMD5(char* src_str)
	{
		std::string dst_mac = "";
		dst_mac.assign(src_str);
		if (dst_mac.empty())
		{
			g_md5[0] = 0;
			return g_md5;
		}

		alva_str_util::replace(dst_mac, ":", "");
		alva_str_util::replace(dst_mac, "-", "");
		dst_mac = alva_str_util::upperCase(dst_mac);

		int str_len = dst_mac.length();

		char *dst_str = NULL;

		if (str_len<=32)
		{
			int dst_len = 32;

			dst_str = new char[dst_len + 1];
			memset(dst_str, 0, dst_len + 1);

			memcpy(dst_str, HEADER_KEY, 32);

			for (int i = 0; i < str_len; i++)
			{
				dst_str[i] = dst_mac[i] ^ HEADER_KEY[i];
				if (dst_str[i] == 0)
				{
					dst_str[i] = 48;
				}
			}

			
		} 
		else
		{
			dst_str = new char[str_len + 1];
			memset(dst_str, 0, str_len + 1);
			memcpy(dst_str, dst_mac.c_str(), str_len);

			for (int i = 0; i < 32; i++)
			{
				dst_str[i] = dst_mac[i] ^ HEADER_KEY[i];
				if (dst_str[i] == 0)
				{
					dst_str[i] = 48;
				}
			}
		}


		int i;
		char enc_str[16] = { 0 };
		MD5_CTX md5;
		MD5Init_local(&md5);
		MD5Update_local(&md5, (unsigned char*)dst_str, str_len);
		MD5Final_local(&md5, (unsigned char*)enc_str);

		std::string dst_md5 = alva_str_util::hextostr(enc_str, 16);

		for (int i=0;i<32;i++)
		{
			g_md5[i] = dst_md5[i];
		}
		g_md5[32] = 0;

		delete[]dst_str;
		return g_md5;
	}

	inline char*  GetDevID_MD5()
	{
		std::string dev_id;
		int ret = GetDevID(dev_id);
		if (-1==ret)
		{
			return NULL;
		}

#ifdef __LOG
		//dev_id = "F02F74F5EDB9";
		//printf(dev_id.c_str());
		//printf("\n");
#endif // __LOG


		return GetMD5((char*)dev_id.c_str());
	}

	inline int CheckTimeValid(std::string time_str)
	{
		if ("" == time_str) return 1;

		time_t dst_t = alva_str_util::hextoint64(time_str);
		tm_tp dst_tt = *(tm_tp*)localtime(&dst_t);
		dst_tt.tm_hour = 23;
		dst_tt.tm_min = 59;
		dst_tt.tm_sec = 59;

		time_t new_dst_t = mktime((struct tm*)&dst_tt);

		time_t cur_t = time(NULL);
		tm_tp cur_tt = *(tm_tp*)localtime(&cur_t);

		cur_tt.tm_hour = 23;
		cur_tt.tm_min = 59;
		cur_tt.tm_sec = 59;

		time_t new_cur_t = mktime((struct tm*)&cur_tt);
		if (new_dst_t>=new_cur_t)
		{
			return 1;
		}

		return -1;

	}

inline int CheckTimeValid_jubensha(std::string time_str)
{
	if ("" == time_str) return 1;

	time_t days_t = (alva_str_util::hextoint64(time_str)+1) *24 * 60 * 60 - 1;

	// begin_t(2023,10,1,0,0,0)
	// time_t begin_time_t = 1699254241;
    time_t begin_time_t = 1696089600;

	time_t new_dst_t = begin_time_t + days_t;

	time_t cur_t = time(NULL);
	tm_tp cur_tt = *(tm_tp*)localtime(&cur_t);

	cur_tt.tm_hour = 23;
	cur_tt.tm_min = 59;
	cur_tt.tm_sec = 59;

	time_t new_cur_t = mktime((struct tm*)&cur_tt);
	if (new_dst_t>=new_cur_t)
	{
		return 1;
	}

	return -1;

}

inline bool startWith_local(const std::string& str, const std::string& head) {
	if (str.size() == 0 || head.size() == 0) return false;

	if (str.size() < head.size()) return false;

	return str.compare(0, head.size(), head) == 0;
}

inline std::vector<std::string> string_split_local(const std::string &str,
									  const std::string &delim) {
	std::vector<std::string> res;
	if ("" == str)
		return res;
	char *strs = new char[str.length() + 1];
	strcpy(strs, str.c_str());

	char *d = new char[delim.length() + 1];
	strcpy(d, delim.c_str());

	char *p = strtok(strs, d);
	while (p) {
		std::string s = p;
		res.push_back(s);
		p = strtok(NULL, d);
	}

	return res;
}


	inline int  GetDevID_MD5_list()
	{
		g_list_dev_md5.clear();
		g_list_mac = get_mac_address_list();

		for (size_t i = 0; i < g_list_mac.size(); i++)
		{
			g_list_dev_md5.push_back(GetMD5((char*)g_list_mac[i].c_str()));
		}

		return g_list_dev_md5.size();
	}


	inline int ValidatDevSN_list(std::string val_md5,int privilege)
	{
		unsigned short enc_code = privilege;

		std::string privilege_str = alva_str_util::hextostr((char*)(&enc_code), 2);
		std::string dst_md5 = "";

		for (size_t n = 0; n < g_list_dev_md5.size(); n++)
		{
			dst_md5 = privilege_str + g_list_dev_md5[n];
			char* src_md5 = GetMD5((char*)dst_md5.c_str());
			//printf("%s   ++++++++++++++++++++++++++++++\n", src_md5);
			//std::cout << "key: ++++++++++++++++++++++++++++++" << src_md5 << '\n';
			int str_len = strlen(src_md5);

			if (str_len > 32) str_len = 32;

			int i = 0;
			while (i < str_len )
			{
				if (src_md5[i] != val_md5[i]) break;
				i++;
			}

			if (i>=str_len)
			{
				return 1;
			} 
		}


		return -1;
	}

	inline int ValidatDevSN_file(std::string fileName, int privilege)
	{
		GetDevID_MD5_list();
		if (g_list_dev_md5.size() == 0)return-1;

		char szLine[50] = { 0 };
		int len = 0;
		int ret = -1;
		int len_dst = g_list_dev_md5[0].length();

		FILE *fp = fopen(fileName.c_str(), "r");
		if (NULL == fp)
		{
			return ValidatDevSN_list(fileName,privilege);
		}

		while (!feof(fp))
		{
		next_line:
			memset(szLine, 0, sizeof(szLine));
			fgets(szLine, sizeof(szLine) - 1, fp); // 包含了\n  

			len = strlen(szLine);
			if (len > 0 && (szLine[len - 1] == '\n' || szLine[len - 1] == '\r'))
			{
				szLine[len - 1] = 0;
				len--;

				if (len > 0 && (szLine[len - 1] == '\n' || szLine[len - 1] == '\r'))
				{
					szLine[len - 1] = 0;
					len--;

				}
			}

			ret = ValidatDevSN_list(szLine, privilege);
			if (1 == ret) break;
		}

		fclose(fp);

		// printf("%d   ++++++++++++++++++++++++++++++\n",ret);

		return ret;
	}



	inline int ValidatDevSN(std::string src_md5)
	{
		char* dev_id = GetDevID_MD5();
		if (NULL == dev_id) return -1;

		char* dst_md5 = GetMD5(dev_id);

		int str_len = strlen(dst_md5);
		if (str_len > 32) str_len = 32;

		for (int i = 0; i < str_len; i++)
		{
			if (src_md5[i] != dst_md5[i]) return -1;
		}

		return 1;
	}

    inline std::string get_DevSN(std::string str_key, int privilege)
    {
        char* dev_sn = GetMD5((char*)str_key.c_str());
        unsigned short enc_code = privilege;

        std::string privilege_str = alva_str_util::hextostr((char*)(&enc_code), 2);

        std::string dst_md5 = privilege_str + dev_sn;
        char* src_md5 = GetMD5((char*)dst_md5.c_str());
        std::string str_sn = src_md5;
        str_sn += privilege_str;

        return str_sn;
    }

    inline void get_fujia_days_str(std::string& fujia_str, std::string& days_str)
	{
		if(startWith_local(fujia_str,"::"))
		{
			std::vector<std::string> res;
			res = string_split_local(fujia_str, "::");
			if(1==res.size())
			{
				days_str=res[0];
			}

			fujia_str ="";


		} else
		{
			std::vector<std::string> res;
			res = string_split_local(fujia_str, ":");
			if(1==res.size())
			{
				fujia_str=res[0];

			} else if (2==res.size())
			{
				fujia_str=res[0];
				days_str=res[1];
			}
		}
	}


	inline int ValidatMD5(std::string src_md5)
	{
		char ip_buf[20] = { 0 };
		char mac_buf[40] = { 0 };

		GetIP_MAC(ip_buf, mac_buf);

		char* dst_md5 = GetMD5(mac_buf);

		int str_len = strlen(dst_md5);
		if (str_len > 32) str_len = 32;

		for (int i = 0; i < str_len; i++)
		{
			if (src_md5[i] != dst_md5[i]) return -1;
		}

		return 1;
	}

	inline int ValidatMD5_file(std::string fileName)
	{
		char ip_buf[20] = { 0 };
		char mac_buf[40] = { 0 };

		GetIP_MAC(ip_buf, mac_buf);

		char* src_md5 = GetMD5(mac_buf);
		int len_dst = strlen(src_md5);

		char szLine[50] = { 0 };
		int len = 0;
		int ret = -1;

		FILE *fp = fopen(fileName.c_str(), "r");
		if (NULL == fp)
		{
			// printf("failed to open dos.txt\n");
			return -1;
		}

		while (!feof(fp))
		{
		next_line:
			memset(szLine, 0, sizeof(szLine));
			fgets(szLine, sizeof(szLine) - 1, fp); 

			len = strlen(szLine);
			if (len > 0 && (szLine[len - 1] == '\n' || szLine[len - 1] == '\r'))
			{
				szLine[len - 1] = 0;
				len--;

				if (len > 0 && (szLine[len - 1] == '\n' || szLine[len - 1] == '\r'))
				{
					szLine[len - 1] = 0;
					len--;

				}
			}

			for (int i = 0; i < len; i++)
			{
				if (src_md5[i] != szLine[i]) goto next_line;
			}

			if (len== len_dst) ret = 1;

			break;
		}

		fclose(fp);

		return ret;
	}



// 在内存中解密 encode_file 生成的格式数据
// merge_buf: 指向 .dat 文件内容的指针
// merge_len: 数据总长度（应 >= 8 + 32）
// mac, encode_key: 用于校验 sn 并生成解密密钥
// data_buf: 输出解密后的数据
// 返回: 0 成功, -1 数据格式错误/长度不足, -2 sn校验失败

int decode_buf(const char* merge_buf, unsigned long long merge_len, std::string mac, std::string encode_key, std::vector<char>& data_buf)
{
    if (NULL == merge_buf || merge_len < sizeof(unsigned long long) + HEADER_LEN) return -1;

    const char* p = merge_buf;

    // 1. 解析 data_len (8字节)
    unsigned long long data_len = *(unsigned long long*)p;
    p += sizeof(unsigned long long);

    if (data_len < HEADER_LEN || HEADER_LEN + (data_len - HEADER_LEN) > merge_len - sizeof(unsigned long long)) {
        return -1;  // data_len 不合法
    }

    // 2. 读取 sn (32字节 hex字符串)
    char sn[HEADER_LEN + 1] = { 0 };
    memcpy(sn, p, HEADER_LEN);
    sn[HEADER_LEN] = '\0';
    p += HEADER_LEN;

    // 3. 校验 sn: 重新计算并与数据中的 sn 对比
    std::string mac_key = mac + encode_key;
    std::vector<char> key_buf(mac_key.begin(), mac_key.end());
    key_buf.push_back('\0');
    char* expected_sn = GetMD5(key_buf.data());
    if (expected_sn[0] == 0) return -1;

    for (int i = 0; i < HEADER_LEN; i++) {
        if (sn[i] != expected_sn[i]) {
            return -2;  // sn 校验失败: mac 或 encode_key 不匹配
        }
    }

    // 4. 计算加密内容大小
    unsigned long long content_len = data_len - HEADER_LEN;

    // 5. 重新生成组合密钥: HEADERMERGE ^ sn
    unsigned char combined_key[HEADER_LEN];
    for (int i = 0; i < HEADER_LEN; i++) {
        combined_key[i] = HEADERMERGE[i] ^ (unsigned char)sn[i];
    }

    // 6. 解密到输出缓冲区: 每字节 -= 组合密钥对应字节
    data_buf.clear();
    data_buf.resize(content_len);
    for (unsigned long long i = 0; i < content_len; i++) {
        data_buf[i] = ((unsigned char)p[i] - combined_key[i % HEADER_LEN]) & 0xFF;
    }

    return 0;
}


// 解密 encode_file 生成的 .dat 文件，还原为原始文件
int decode_file(std::string dat_file, std::string mac, std::string encode_key)
{
    // 1. 读取整个 .dat 文件到内存
    FILE* fp_src = fopen(dat_file.c_str(), "rb");
    if (NULL == fp_src) return -1;

    fseek(fp_src, 0, SEEK_END);
    unsigned long long file_len = ftell(fp_src);
    fseek(fp_src, 0, SEEK_SET);

    std::vector<char> file_buf(file_len);
    fread(file_buf.data(), 1, file_len, fp_src);
    fclose(fp_src);

    // 2. 调用内存解密
    std::vector<char> data_buf;
    int ret = decode_buf(file_buf.data(), file_len, mac, encode_key, data_buf);
    if (ret != 0) return ret;

    // 3. 生成输出文件名: 去掉 .dat 后缀，否则加 _decoded
    std::string out_file = dat_file;
    std::string dat_suffix = ".dat";
    if (out_file.length() >= dat_suffix.length() &&
        out_file.compare(out_file.length() - dat_suffix.length(),
                         dat_suffix.length(), dat_suffix) == 0) {
        out_file = out_file.substr(0, out_file.length() - dat_suffix.length());
    } else {
        out_file = dat_file + "_decoded";
    }

    FILE* fp_dst = fopen(out_file.c_str(), "wb");
    if (NULL == fp_dst) {
        return -1;
    }

    fwrite(data_buf.data(), 1, data_buf.size(), fp_dst);
    fclose(fp_dst);

    return 0;
}




#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
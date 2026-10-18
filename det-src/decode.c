#include "stddef.h"

const unsigned int HEADER_LEN = 32;

unsigned char  HEADERBYTE[] =
{ 0x98, 0x58, 0xba, 0x29, 0xbd, 0x51, 0x48, 0xdd, 0xbc, 0xfe, 0xaa, 0x61, 0x61, 0xd2, 0x83, 0x7a,
	0x58, 0xa2, 0x81, 0xf3, 0xcd, 0x67, 0x41, 0xea, 0x9f, 0xf1, 0x7c, 0x63, 0x34, 0xad, 0xe5, 0x80 };

unsigned char  HEADERMERGE[] = { 0xb2, 0x29, 0xa7, 0xdb, 0x40, 0x51, 0x40, 0x42, 0x9f, 0xbc, 0xff, 0xf2, 0xc0, 0x89, 0xfd, 0x97,
0xA0, 0x41, 0x88, 0x26, 0xCE, 0x87, 0x4D, 0x79, 0x84, 0x5e, 0x7b, 0xc5, 0x5c, 0xb, 0xab, 0xd8 };


int en_merge(unsigned char *pSrc, unsigned long src_len, int encode)
{
	unsigned char* pData = pSrc;
	unsigned char* pEn = HEADERMERGE;

	unsigned long i = 0;
	unsigned long handle_len = src_len;

	while (i < src_len)
	{
		if (1 == encode)
		{
			*pData = (*pData + *pEn) & 0xFF;
		}
		else
		{
			*pData = (*pData - *pEn) & 0xFF;
		}

		pData++;
		pEn++;
		i++;

		if (i%HEADER_LEN == 0)
		{
			pEn = HEADERMERGE;
		}
	}

	return src_len;
}

int  ReadMergeMem(char *merge_buf, int merge_len, int idx, char** data, unsigned long long* data_size)
{
	if (idx < 0) return -1;

	int size_bytes = sizeof(unsigned long long);

	unsigned long long src_len = 0;
	char *src_buf = NULL;
	unsigned char need_decode = 0;

    char *pBuf = merge_buf;
    char *pEnd = merge_buf+merge_len;

	// 循环找到 idx 所在文件块
	while (idx>=0 && pBuf<pEnd)
	{
		// 获取文件大小
        src_len = *((unsigned long long*)pBuf);
        pBuf+=size_bytes;

        need_decode = *((unsigned char*)pBuf);
        pBuf+=1;

		if (0==idx)
		{
			break;
		} 
		else
		{
            pBuf+=src_len;
			idx--;
		}

	}

	if (pBuf>= pEnd)
	{
		*data = NULL;
		*data_size = 0;

		return 0;
	}	

	if (1 == need_decode)
	{
		en_merge(pBuf, src_len, 0);
	}

	*data = pBuf;
	*data_size = src_len;

	return src_len;
}

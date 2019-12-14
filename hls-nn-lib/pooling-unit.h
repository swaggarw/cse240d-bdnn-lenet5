#pragma once
#include <hls_stream.h>
#include <ap_int.h>
using namespace hls;
#include <iostream>
using namespace std;
#include <assert.h>

// #define POOL_DEBUG

template <	unsigned NumVecs,
			unsigned Ibit,
			unsigned K,
			unsigned Cin,
			unsigned InP>
void POOL(
	stream<ap_uint<InP*Cin*Ibit> >& vec,
	stream<ap_uint<Cin*Ibit> >& out,
	const unsigned reps = 1)
{
	static_assert( (K*K)%InP == 0, "K*K mod InP is not 0" );

	ap_uint<Cin*Ibit> result;
	unsigned wVec = 0;

	for (unsigned rep = 0; rep < reps*NumVecs*(K*K)/InP; rep++) {
#pragma HLS PIPELINE II=1

		if (wVec == 0)
			result = 0;

		ap_uint<InP*Cin*Ibit> tempVec = vec.read();

		for (unsigned c = 0; c < Cin; c++) {
#pragma HLS UNROLL
			for (unsigned p = 0; p < InP; p++) {
				ap_uint<Ibit> temp = tempVec( (p*Cin+c+1)*Ibit-1 , (p*Cin+c)*Ibit );
				
				result( (c+1)*Ibit-1, c*Ibit ) = (temp > result( (c+1)*Ibit-1, c*Ibit )) ? temp : result( (c+1)*Ibit-1, c*Ibit );
			}
		}

		if (wVec == (K*K)/InP-1)
			out.write(result);
		
		if (wVec == (K*K)/InP-1)
			wVec = 0;
		else
			wVec++;
	}
}

template <	unsigned Ibit,
			unsigned K,
			unsigned Cin,
			unsigned InP>
void POOL_variable(
	stream<ap_uint<InP*Cin*Ibit> >& vec,
	stream<ap_uint<Cin*Ibit> >& out,
	const unsigned NumVecs,
	const unsigned reps = 1)
{
	static_assert( (K*K)%InP == 0, "K*K mod InP is not 0" );

	ap_uint<Cin*Ibit> result;
	unsigned wVec = 0;

	for (unsigned rep = 0; rep < reps*NumVecs*(K*K)/InP; rep++) {
#pragma HLS PIPELINE II=1

		if (wVec == 0)
			result = 0;

		ap_uint<InP*Cin*Ibit> tempVec = vec.read();

		for (unsigned c = 0; c < Cin; c++) {
#pragma HLS UNROLL
			for (unsigned p = 0; p < InP; p++) {
				ap_uint<Ibit> temp = tempVec( (p*Cin+c+1)*Ibit-1 , (p*Cin+c)*Ibit );
				
				result( (c+1)*Ibit-1, c*Ibit ) = (temp > result( (c+1)*Ibit-1, c*Ibit )) ? temp : result( (c+1)*Ibit-1, c*Ibit );
			}
		}

		if (wVec == (K*K)/InP-1)
			out.write(result);
		
		if (wVec == (K*K)/InP-1)
			wVec = 0;
		else
			wVec++;
	}
}

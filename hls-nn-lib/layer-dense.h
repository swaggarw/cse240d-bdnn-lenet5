#pragma once
#include <hls_stream.h>
#include <ap_int.h>
using namespace hls;
#include <iostream>
using namespace std;
#include <assert.h>

#include "matrix-vector-unit.h"

template <	unsigned Din,
			unsigned Dout,
			unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned Abit,
			unsigned InP,
			unsigned OutP,
			unsigned ScaleBits,
			unsigned FactorScaleBits>
void DENSE_ACT(
	stream<ap_uint<InP*Ibit> >& in,
	const ap_uint<InP*Wbit> weights[OutP][(Din/InP)*(Dout/OutP)],
	const ap_int<Mbit> factorA[OutP][Dout/OutP],
	const ap_int<Mbit> factorB[OutP][Dout/OutP],
	stream<ap_uint<OutP*Abit> >& out,
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	MVAU_rowfirst<1, Ibit, Wbit, Mbit, Abit, Din, Dout, InP, OutP, ScaleBits, FactorScaleBits>(in, weights, factorA, factorB, out, reps);
}

template <	unsigned Din,
			unsigned Dout,
			unsigned Ibit,
			unsigned Wbit,
			unsigned Mbit,
			unsigned InP,
			unsigned OutP,
			unsigned ScaleBits>
void DENSE_NOACT(
	stream<ap_uint<InP*Ibit> >& in,
	const ap_uint<InP*Wbit> weights[OutP][(Din/InP)*(Dout/OutP)],
	stream<ap_uint<OutP*Mbit> >& out,
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW

	MVU_rowfirst<1, Ibit, Wbit, Mbit, Din, Dout, InP, OutP>(in, weights, out, reps);
}

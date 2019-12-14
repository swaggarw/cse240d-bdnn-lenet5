#define TESTBENCH
// #define DEBUG

#define AP_INT_MAX_W 16384

#include "hls-nn-lib.h"
#include "../training/mnist-cnn-config.h"
#include "../training/mnist-cnn-params.h"

void DoCompute(stream<ap_axis >& in, stream<ap_axis >& out, unsigned int numReps) {
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS RESOURCE variable=weights0 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights0 complete dim=0
#pragma HLS RESOURCE variable=factorA0 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB0 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=factorA0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB0 complete dim=0
#pragma HLS RESOURCE variable=weights1 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights1 complete dim=0
#pragma HLS RESOURCE variable=factorA1 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB1 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=factorA1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB1 complete dim=0
#pragma HLS RESOURCE variable=weights2 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights2 complete dim=0
#pragma HLS RESOURCE variable=factorA2 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB2 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=factorA2 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB2 complete dim=0
#pragma HLS RESOURCE variable=weights3 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights3 complete dim=0
#pragma HLS RESOURCE variable=factorA3 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB3 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=factorA3 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB3 complete dim=0
#pragma HLS RESOURCE variable=weights4 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=weights4 complete dim=0
#pragma HLS RESOURCE variable=factorA4 core=RAM_1P_BRAM
#pragma HLS RESOURCE variable=factorB4 core=RAM_1P_BRAM
#pragma HLS ARRAY_PARTITION variable=factorA4 complete dim=0
#pragma HLS ARRAY_PARTITION variable=factorB4 complete dim=0

#pragma HLS DATAFLOW

	// 2 rows of 28 row image per line -> per image 14 lines
	const unsigned int NumLinesPerRep = 14;

	stream<ap_uint<448> > in_stream_extract("in_stream_extract");
	ExtractPixels<448, NumLinesPerRep> (in, in_stream_extract, numReps);

	stream<ap_uint<L0_Cin*L0_Ibit> > in_stream("in_stream");
	ReduceWidth<448, L0_Cin*L0_Ibit, NumLinesPerRep> (in_stream_extract, in_stream, numReps);

#ifdef DEBUG
	Monitor<L0_Din, L0_Cin, L0_Ibit>(in_stream, (char*)"./log/mon_in_stream.log", numReps);
#endif

stream<ap_uint<L0_Cout*L0_Abit> > conv0("conv0");
CONV2D_ACT_NoP<L0_K, L0_S, L0_Din, L0_Cin, L0_Cout, L0_Ibit, L0_Wbit, L0_Mbit, L0_Abit, L0_MVTU_InP, L0_MVTU_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
(in_stream, weights0, factorA0, factorB0, conv0, numReps);
#ifdef DEBUG
Monitor<L0_Din/L0_S, L0_Cout, L0_Abit>(conv0, (char*)"log/mon_conv0.log", numReps);
#endif

stream<ap_uint<L5_Cin*L5_Ibit> > pool0("pool0");
POOL2D_NoP<L5_K, L5_S, L5_Din, L5_Cin, L5_Ibit>
(conv0, pool0, numReps);
#ifdef DEBUG
Monitor<L5_Din/L5_S, L5_Cin, L5_Ibit>(pool0, (char*)"log/mon_pool0.log", numReps);
#endif

stream<ap_uint<L1_Cout*L1_Abit> > conv1("conv1");
CONV2D_ACT_NoP<L1_K, L1_S, L1_Din, L1_Cin, L1_Cout, L1_Ibit, L1_Wbit, L1_Mbit, L1_Abit, L1_MVTU_InP, L1_MVTU_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
(pool0, weights1, factorA1, factorB1, conv1, numReps);
#ifdef DEBUG
Monitor<L1_Din/L1_S, L1_Cout, L1_Abit>(conv1, (char*)"log/mon_conv1.log", numReps);
#endif

stream<ap_uint<L6_Cin*L6_Ibit> > pool1("pool1");
POOL2D_NoP<L6_K, L6_S, L6_Din, L6_Cin, L6_Ibit>
(conv1, pool1, numReps);
#ifdef DEBUG
Monitor<L6_Din/L6_S, L6_Cin, L6_Ibit>(pool1, (char*)"log/mon_pool1.log", numReps);
#endif

stream<ap_uint<L2_OutP*L2_Abit> > dense0("dense0");
DENSE_ACT<L2_Din, L2_Dout, L2_Ibit, L2_Wbit, L2_Mbit, L2_Abit, L2_InP, L2_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
(pool1, weights2, factorA2, factorB2, dense0, numReps);

stream<ap_uint<L3_OutP*L3_Abit> > dense1("dense1");
DENSE_ACT<L3_Din, L3_Dout, L3_Ibit, L3_Wbit, L3_Mbit, L3_Abit, L3_InP, L3_OutP, SCALE_BITS, FACTOR_SCALE_BITS>
(dense0, weights3, factorA3, factorB3, dense1, numReps);

stream<ap_uint<L4_OutP*L4_Mbit> > dense2("dense2");
DENSE_NOACT<L4_Din, L4_Dout, L4_Ibit, L4_Wbit, L4_Mbit, L4_InP, L4_OutP, SCALE_BITS>
(dense1, weights4, dense2, numReps);

	stream<ap_uint<512> > out_nolast("out_nolast");
	AppendZeros<10*L4_Mbit, 512, 1> (dense2, out_nolast, numReps);

	AddLast<1>(out_nolast, out, numReps);
}

void ArbitrateCompute(stream<ap_axis >& in, stream<ap_axis >& out, unsigned int numReps) {
	numReps = 100;
	stream<ap_axis> instream1;
	stream<ap_axis> instream2;

	stream<ap_axis> outstream1;
	stream<ap_axis> outstream2;

#pragma HLS stream variable=instream1 depth=577
#pragma HLS stream variable=instream2 depth=577
#pragma HLS stream variable=outstream1 depth=577
#pragma HLS stream variable=outstream2 depth=577
	unsigned int i = 0;
	unsigned int j = 0;

//#pragma HLS UNROLL factor=10
input_arbitration:	for(i = 0; i < numReps; i++) {
#pragma HLS PIPELINE II=1
		for(j = 0; j < 14; j++) {
			if(i % 2 == 0) {
				instream1.write(in.read());
			}
			else {
				instream2.write(in.read());
			}
		}
	}

	DoCompute(instream1, outstream1, (numReps+1)/2);
	DoCompute(instream2, outstream2, (numReps)/2);

output_combination:	for(i = 0; i < numReps; i++) {
#pragma HLS PIPELINE II=1
		if(i % 2 == 0) {
			out.write(outstream1.read());
		}
		else {
			out.write(outstream2.read());
		}
	}
}

// TESTBENCH
//#ifdef TESTBENCH
//#include "loader.h"
//#include "stdio.h"
//
//using namespace std;
//
//int main(int argc, char* argv[]) {
//// #define NUM_SAMPLES 50
//	char* pathToDataset;
//	unsigned int NUM_SAMPLES = 0;
//	if (argc != 3) {
//		cout << "Usage: ./t <pathToDataset>" << endl;
//		return 0;
//	}
//	else {
//		pathToDataset = argv[1];
//		NUM_SAMPLES=std::atoi(argv[2]);
//}
//
//	loader load = loader();
//
//	load.load_libsvm_data(pathToDataset, NUM_SAMPLES, 784, 10);
//	load.x_normalize(0, 'r');
//
//	// One line will contain 448 useful bits
//	const unsigned int data_points_per_line = 448/L0_Ibit;
//	// Per image, we need 14 lines
//	const unsigned int buffer_size = NUM_SAMPLES*14;
//	stream<ap_axis > inputStream;
//
//	unsigned int index = 0;
//	for (unsigned int i = 0; i < buffer_size; i++) {
//		ap_axis temp;
//		for (unsigned int j = 0; j < data_points_per_line; j++) {
//			temp.data( L0_Ibit*(j+1)-1, L0_Ibit*j ) = ((unsigned int)(load.x[index++]*255.0));
//		}
//		cout << "inputStream[" << i << "]: " << hex << temp.data << dec << endl;
//
//		inputStream.write(temp);
//	}
//
//	stream<ap_axis > outputStream;
//
//	ArbitrateCompute(inputStream, outputStream, NUM_SAMPLES);
//
//
//	ap_axis outputBuffer[NUM_SAMPLES];
//
//	for (unsigned int i = 0; i < NUM_SAMPLES; i++) {
//		outputBuffer[i] = outputStream.read();
//	}
//
//	unsigned long MASK = ((long)1 << L4_Mbit) - 1;
//	unsigned int count_trues = 0;
//	for (unsigned int i = 0; i < NUM_SAMPLES; i++) {
//		cout << "At sample: " << i << endl;
//		cout << "outputBuffer[" << i << "]: " << hex << outputBuffer[i].data << dec << endl;
//		int max = 0;
//		unsigned int prediction = -1;
//		for (unsigned int j = 0; j < L4_Dout; j++) {
//			int temp = (outputBuffer[i].data >> (j*L4_Mbit)) & MASK;
//			temp = temp >> SCALE_BITS;
//			cout << "outputBuffer[" << i << "][" << j << "]: " << static_cast<int>(temp) << endl;
//			if (temp > max){
//				max = temp;
//				prediction = j;
//			}
//		}
//		cout << "prediction: " << prediction << endl;
//		if (load.y[i*10 + prediction] == 1)
//			count_trues++;
//	}
//
//	float accuracy = 0;
//
//	cout << "Accuracy : " << (float(count_trues)/float(NUM_SAMPLES))*100 << "%" << endl;
//	// cout << count_trues << " correct out of " << NUM_SAMPLES << endl;
//}
//#endif

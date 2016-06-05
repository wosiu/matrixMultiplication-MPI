#include <unordered_map>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <mpi.h>
#include <assert.h>
#include <getopt.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "densematgen.h"
#include "utils.h"

#define debon true
#define deb(burak) if(debon) {cout<<__LINE__<< " DEB-> "<<#burak<<": "<<burak<<endl;}
#define debv(burak) if(debon) {cout<<__LINE__<< " DEB-> "<<#burak<<": \t"; for(unsigned int zyx=0;zyx<burak.size();zyx++) cout<<burak[zyx]<<" "; cout<<endl;}
#define debt(burak,SIzE) if(debon) {cout<<__LINE__<< " DEB-> "<<#burak<<": \t"; for(unsigned int zyx=0;zyx<SIzE;zyx++) cout<<burak[zyx]<<" "; cout<<endl;}
#define debend if(debon) {cout<<"_____________________"<<endl;}

using namespace std;

//typedef void* sparse_type;
/**
 * x - rows, y - columns
 */
struct cell {
	int x, y;
	double v;
};
//typedef pair<pair<int,int>, double> cell;
typedef vector<cell> sparse_type;
class Matrix {
};

class SparseMatrix : Matrix {
public:
	sparse_type cells;
	int ncol, nrow;
};


void commitSparseCellType() {
	const int nitems = 3;
	int blocklengths[nitems] = { 1, 1, 1 };
	MPI_Datatype types[nitems] = { MPI_INT, MPI_INT, MPI_DOUBLE };
	MPI_Datatype mpi_type;
	MPI_Aint offsets[nitems];

	offsets[0] = offsetof(cell, x);
	offsets[1] = offsetof(cell, y);
	offsets[2] = offsetof(cell, v);

	MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_type);
	MPI_Type_commit(&mpi_type);
}

sparse_type readCSR(FILE * stream) {
	int row_num, col_num, nz_num, max_num_nz_row;
	if ( fscanf(stream, "%d %d %d %d ", &row_num, &col_num, &nz_num, &max_num_nz_row) != 4 ) {
		throw runtime_error("Wrong format of sparse matrix - header");
	}
	vector<cell> matrix(nz_num);

	for (int i = 0; i < nz_num; i++) {
		if ( fscanf(stream, "%lf ", &matrix[i].v) != 1) {
			throw runtime_error("Wrong format of sparse matrix - values");
		}
	}

	int cell_i = 0;
	int offset;
	if (fscanf(stream, "%d ", &offset) != 1) { // omit
		throw runtime_error("Wrong format of sparse matrix - offset omit");
	}
	for (int row_i = 0; row_i < row_num; row_i++) {
		if ( fscanf(stream, "%d ", &offset) != 1 ) {
			throw runtime_error("Wrong format of sparse matrix - offset");
		}
		for (; cell_i < offset; cell_i++) {
			matrix[cell_i].x = row_i;
		}
	}

	// JA
	for (int i = 0; i < nz_num; i++) {
		if ( fscanf(stream, "%d ", &matrix[i].y) != 1 ) {
			throw runtime_error("Wrong format of sparse matrix - columns");
		}
	}

	if (!feof(stream)) {
		throw runtime_error("Wrong format of sparse matrix - some data unread.");
	}
	return matrix;
}

void printSparse(const sparse_type& matrix) {
	for (cell ce : matrix) {
		printf("%d, %d: %lf\n", ce.x, ce.y, ce.v);
	}
}

bool isSparseInited(const sparse_type& sparse) {
	return !sparse.empty();
}

vector<sparse_type> colChunks(const sparse_type &sparse, int n) {
	vector<sparse_type> chunks(n);
	map<int, sparse_type> columns;
	//vector<int> columns;
	//for_each(sparse.begin(), sparse.end(), [&](cell& e) { columns.push_back(e.y); });
	//sort(sparse.begin(), sparse.end(), byYcmp) {}
	for (auto& a : sparse) {
		columns[a.y].push_back(a);
	}
	auto nzColNum = columns.size();

	int base = nzColNum / n; // base number of columns for each process
	int add = nzColNum % n; // number of processes with +1 column than the rest
	auto cit = columns.begin();

	for (int chunk_id = 0; chunk_id < n; chunk_id++) {
		int total_col = base + int(chunk_id < add);
		for (int count = 0; count < total_col; count++) {
			auto& chunk = cit->second;
			chunks[chunk_id].insert(chunks[chunk_id].end(), chunk.begin(), chunk.end());
			cit++;
		}
	}

	return chunks;
}


int main(int argc, char * argv[]) {
	if (true) {
		/*FILE * pFile = fopen("../sparse1.in", "r");
		auto tmp = readCSR(pFile);
		printSparse(tmp);
		return 0;*/
	}

	int show_results = 0;
	int use_inner = 0;
	int gen_seed = -1;
	int repl_fact = 1;

	int option = -1;

	double comm_start = 0, comm_end = 0, comp_start = 0, comp_end = 0;
	int num_processes = 1;
	int mpi_rank = 0;
	int exponent = 1;
	double ge_element = 0;
	int count_ge = 0;

	sparse_type sparse;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

	while ((option = getopt(argc, argv, "vis:f:c:e:g:")) != -1) {
		switch (option) {
			case 'v':
				show_results = 1;
				break;
			case 'i':
				use_inner = 1;
				break;
			case 'f':
				if ((mpi_rank) == 0) {
					FILE * pFile = fopen(optarg, "r");
					if (pFile == NULL) {
						fprintf(stderr, "Cannot open sparse matrix file %s\n", optarg);
						MPI_Finalize();
						return 3;
					}
					// Process 0 reads the CSR sparse matrix
					try {
						sparse_type sparse = readCSR(pFile);
					} catch (runtime_error const& e) {
						fprintf(stderr, "%s : %s", e.what(), optarg);
						MPI_Finalize();
						return 3;
					}
					fclose(pFile);
				}
				break;
			case 'c':
				repl_fact = atoi(optarg);
				break;
			case 's':
				gen_seed = atoi(optarg);
				break;
			case 'e':
				exponent = atoi(optarg);
				break;
			case 'g':
				count_ge = 1;
				ge_element = atof(optarg);
				break;
			default:
				fprintf(stderr, "error parsing argument %c exiting\n", option);
				MPI_Finalize();
				return 3;
		}
	}
	if ((gen_seed == -1) || ((mpi_rank == 0) && isSparseInited(sparse))) {
		fprintf(stderr, "error: missing seed or sparse matrix file; exiting\n");
		MPI_Finalize();
		return 3;
	}

	// init MPI
	commitSparseCellType();

	// prepare sparse chunks to scatter
	vector<sparse_type> sparse_chunks;
	if ((mpi_rank) == 0) {
		sparse_chunks = colChunks(sparse, num_processes);
	} else {
		// TODO
	}

	comm_start = MPI_Wtime();
// FIXME: scatter sparse matrix; cache sparse matrix; cache dense matrix
	if ((mpi_rank) == 0) {
//		scatterSparse(sparse);
	}
//	MPI_Gather(&my_sparse, 1, MPI_DOUBLE, sub_avgs, 1, MPI_FLOAT, 0,
//				MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	comm_end = MPI_Wtime();

	comp_start = MPI_Wtime();
// compute C = A ( A ... (AB ) )
// exchange sparse columns within columns group
// (p_1, ..., p_c); (p_c+1, ..., p_2*c); ...
// TODO

// dim(dense) == dim(result)
	for (int exp_i = 0; exp_i < exponent; ++exp_i) {
//		partialRes(my_sparse, my_dense, &my_result);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	comp_end = MPI_Wtime();

	if (show_results) {
		// FIXME: replace the following line: print the whole result matrix
		printf("1 1\n42\n");
	}
	if (count_ge) {
		// FIXME: replace the following line: count ge elements
		printf("54\n");
	}

	MPI_Finalize();

	return 0;
}

//void tmp(result, sparse, dense) {
//
//}

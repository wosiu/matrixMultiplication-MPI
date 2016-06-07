#include <unordered_map>
#include <iostream>
#include <string>
#include <fstream>      // std::ifstream
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <mpi.h>
#include <cassert>
#include <getopt.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
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

ostream & operator <<(std::ostream & s, const cell & v) {
	return s << '(' << v.x << ',' << v.y << ":" << v.v << ")";
}

//typedef pair<pair<int,int>, double> cell;
typedef vector<cell> sparse_type;
class Matrix {
public:
	int nrow, ncol;
};

class SparseMatrix: public Matrix {
public:
	sparse_type cells;

	void print() const {
		for (cell ce : cells) {
			printf("%d, %d: %lf\n", ce.x, ce.y, ce.v);
		}
	}

	bool isEmpty() const {
		return cells.empty();
	}

};

MPI_Datatype commitSparseCellType() {
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
	return mpi_type;
}

SparseMatrix readCSR(ifstream & stream) {
	ios::sync_with_stdio(false);
	int row_num, col_num, nz_num, max_num_nz_row;

	if (!(stream >> row_num >> col_num >> nz_num >> max_num_nz_row).good()) {
		throw runtime_error("Wrong format of sparse matrix - header");
	}
	vector<cell> cells(nz_num);

	for (int i = 0; i < nz_num; i++) {
		if (!(stream >> cells[i].v).good()) {
			throw runtime_error("Wrong format of sparse matrix - values");
		}
	}

	int cell_i = 0;
	int offset;

	if (!(stream >> offset).good()) { // omit
		throw runtime_error("Wrong format of sparse matrix - offset omit");
	}
	for (int row_i = 0; row_i < row_num; row_i++) {
		if (!(stream >> offset).good()) {
			throw runtime_error("Wrong format of sparse matrix - offset");
		}
		for (; cell_i < offset; cell_i++) {
			cells[cell_i].x = row_i;
		}
	}

	// JA
	for (int i = 0; i < nz_num; i++) {
		if (!(stream >> cells[i].y).good()) {
			throw runtime_error("Wrong format of sparse matrix - columns");
		}
	}

	char ch = 0;
	if (!stream.eof()) {
		stream >> ch;
	}
	if (ch != 0 || !stream.eof()) {
		throw runtime_error("Wrong format of sparse matrix - some data unread, check if no EOL at the end of file.");
	}

	SparseMatrix matrix;
	matrix.cells = cells;
	matrix.nrow = row_num;
	matrix.ncol = col_num;
	return matrix;
}

SparseMatrix readCSR(FILE * stream) {
	int row_num, col_num, nz_num, max_num_nz_row;
	if (fscanf(stream, "%d %d %d %d ", &row_num, &col_num, &nz_num, &max_num_nz_row) != 4) {
		throw runtime_error("Wrong format of sparse matrix - header");
	}
	vector<cell> cells(nz_num);

	for (int i = 0; i < nz_num; i++) {
		if (fscanf(stream, "%lf ", &cells[i].v) != 1) {
			throw runtime_error("Wrong format of sparse matrix - values");
		}
	}

	int cell_i = 0;
	int offset;
	if (fscanf(stream, "%d ", &offset) != 1) { // omit
		throw runtime_error("Wrong format of sparse matrix - offset omit");
	}
	for (int row_i = 0; row_i < row_num; row_i++) {
		if (fscanf(stream, "%d ", &offset) != 1) {
			throw runtime_error("Wrong format of sparse matrix - offset");
		}
		for (; cell_i < offset; cell_i++) {
			cells[cell_i].x = row_i;
		}
	}

	// JA
	for (int i = 0; i < nz_num; i++) {
		if (fscanf(stream, "%d ", &cells[i].y) != 1) {
			throw runtime_error("Wrong format of sparse matrix - columns");
		}
	}

	if (!feof(stream)) {
		throw runtime_error("Wrong format of sparse matrix - some data unread.");
	}
	SparseMatrix matrix;
	matrix.cells = cells;
	matrix.nrow = row_num;
	matrix.ncol = col_num;
	return matrix;
}

SparseMatrix readCSR(char* path) {
//	FILE * pFile = fopen(optarg, "r");
//	if (pFile == NULL) {
//		throw runtime_error("Cannot open sparse matrix file " + path);
//	}
//	SparseMatrix sparse = readCSR(pFile);
//	fclose(pFile);

	ifstream txtFile;
	txtFile.open(path, ifstream::in);
	if (!txtFile.is_open()) {
		throw runtime_error("Cannot open sparse matrix file " + string(path));
	}
	SparseMatrix sparse = readCSR(txtFile);
	txtFile.close();
	return sparse;
}

vector<sparse_type> colChunks(const SparseMatrix &sparse, int n) {
	vector<sparse_type> chunks(n);
	map<int, sparse_type> vectors;

	// TODO przerobic na wersje column albo row chunks, dobierac sie do klucza po offsecie struktury, MARGIN = 0/1

	for (auto& a : sparse.cells) {
		vectors[a.y].push_back(a);
	}
	auto nzColNum = vectors.size();

	int base = nzColNum / n; // base number of columns for each process
	int add = nzColNum % n; // number of processes with +1 column than the rest
	auto cit = vectors.begin();

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


sparse_type sparseIsendIrecv(const SparseMatrix& sparse, MPI_Datatype MPI_SPARSE_CELL, int num_processes, int mpi_rank) {
	sparse_type my_sparse_chunk;
	if ((mpi_rank) == 0) {
		// prepare sparse chunks to scatter
		vector<sparse_type> sparse_chunks;
		sparse_chunks = colChunks(sparse, num_processes);
		my_sparse_chunk = sparse_chunks[0];

		for (int proc = 1; proc < num_processes; proc++) {
			cell * data = sparse_chunks[proc].data();
//			deb(proc); debt(data, sparse_chunks[proc].size());
			MPI_Request req;
			MPI_Isend(data, sparse_chunks[proc].size(), MPI_SPARSE_CELL, proc, 0, MPI_COMM_WORLD, &req);
			MPI_Request_free(&req);
		}
	} else {
		MPI_Request req;
		MPI_Status status;
		int incoming_size;

		MPI_Probe(0, 0, MPI::COMM_WORLD, &status);
		MPI_Get_count(&status, MPI_SPARSE_CELL, &incoming_size);
		deb(mpi_rank); deb(incoming_size);
		my_sparse_chunk.resize(incoming_size);
		MPI_Irecv(my_sparse_chunk.data(), incoming_size, MPI_SPARSE_CELL, 0, 0, MPI_COMM_WORLD, &req);
		MPI_Request_free(&req);
	}
	return my_sparse_chunk;
}

sparse_type sparseScatterV(SparseMatrix& sparse, MPI_Datatype MPI_SPARSE_CELL, int num_processes, int mpi_rank) {
	sparse_type my_sparse_chunk;
	int chunks_num = 0;
	vector<sparse_type> sparse_chunks;

	if ((mpi_rank) == 0) {
		// prepare sparse chunks to scatter
		sparse_chunks = colChunks(sparse, num_processes);
		sparse.cells.clear();
		chunks_num = sparse_chunks.size();
	}

	cell* sendbuf = NULL;
	int scounts[chunks_num];
	int displs[chunks_num];

	if ((mpi_rank) == 0) {
		// flatten chunks
		for (int i = 0; i < chunks_num; i++) {
			sparse.cells.insert(sparse.cells.end(), sparse_chunks[i].begin(), sparse_chunks[i].end());
			scounts[i] = sparse_chunks[i].size();
			displs[i] = (i == 0) ? 0 : displs[i-1] + scounts[i-1];
		}
//			deb(proc); debt(data, sparse_chunks[proc].size());
		sendbuf = sparse.cells.data();
//		deb("after")
//		deb(sparse.cells.size());
//		debt(sendbuf, sparse.cells.size());
	}

	// scatter chunks sizes
	int mysize = -1;
	MPI_Scatter(scounts, 1, MPI_INT, &mysize, 1, MPI_INT, 0, MPI_COMM_WORLD );
//	if (debon) cout << mpi_rank << ": mysize " << mysize << endl;
	//	if (debon && sendbuf != NULL) { debt(sendbuf, 4); debt(scounts , num_processes); debt(displs , num_processes); }

	my_sparse_chunk.resize(mysize);

	// scatter data
	MPI_Scatterv(sendbuf, scounts , displs, MPI_SPARSE_CELL,
			my_sparse_chunk.data(), mysize, MPI_SPARSE_CELL, 0, MPI_COMM_WORLD);

//	deb(my_sparse_chunk.size());
//	debv(my_sparse_chunk)
	return my_sparse_chunk;
}

sparse_type sparseScatterV2(SparseMatrix& sparse, MPI_Datatype MPI_SPARSE_CELL, int num_processes, int mpi_rank) {
	sparse_type my_sparse_chunk;

	cell* sendbuf = NULL;
	int scounts[num_processes];
	int displs[num_processes];

	if ((mpi_rank) == 0) {
		int nz = sparse.cells.size();
		sendbuf = sparse.cells.data();
		int base = nz / num_processes; // base number of columns for each process
		int add = nz % num_processes; // number of processes with +1 column than the rest

		for ( int i = 0; i < num_processes; i++ ) {
			scounts[i] = base + int(i < add);
			displs[i] = (i == 0) ? 0 : displs[i-1] + scounts[i-1];
		}
	}

	// scatter chunks sizes
	int mysize = -1;
	MPI_Scatter(scounts, 1, MPI_INT, &mysize, 1, MPI_INT, 0, MPI_COMM_WORLD );
//	if (debon) cout << mpi_rank << ": mysize " << mysize << endl;
	//	if (debon && sendbuf != NULL) { debt(sendbuf, 4); debt(scounts , num_processes); debt(displs , num_processes); }

	my_sparse_chunk.resize(mysize);

	// scatter data
	MPI_Scatterv(sendbuf, scounts , displs, MPI_SPARSE_CELL,
			my_sparse_chunk.data(), mysize, MPI_SPARSE_CELL, 0, MPI_COMM_WORLD);

//	deb(my_sparse_chunk.size());
//	debv(my_sparse_chunk)
	return my_sparse_chunk;
}


sparse_type sparseIsendIrecv2(SparseMatrix& sparse, MPI_Datatype MPI_SPARSE_CELL, int num_processes, int mpi_rank) {
	sparse_type my_sparse_chunk;

	if ((mpi_rank) == 0) {
		// prepare sparse chunks to scatter
		cell * data = sparse.cells.data();
		int nz = sparse.cells.size();
		int base = nz / num_processes; // base number of columns for each process
		int add = nz % num_processes; // number of processes with +1 column than the rest

		int rootCount = base + int(0 < add);
		int offset = rootCount;
		my_sparse_chunk = sparse_type(data, data + rootCount);

		for (int proc = 1; proc < num_processes; proc++) {
			int count = base + int(proc < add);

			MPI_Request req;
			MPI_Isend(data + offset, count, MPI_SPARSE_CELL, proc, 0, MPI_COMM_WORLD, &req);
			MPI_Request_free(&req);
			offset += count;
		}
	} else {
		MPI_Request req;
		MPI_Status status;
		int incoming_size;

		MPI_Probe(0, 0, MPI::COMM_WORLD, &status);
		MPI_Get_count(&status, MPI_SPARSE_CELL, &incoming_size);
		deb(mpi_rank); deb(incoming_size);
		my_sparse_chunk.resize(incoming_size);
		MPI_Irecv(my_sparse_chunk.data(), incoming_size, MPI_SPARSE_CELL, 0, 0, MPI_COMM_WORLD, &req);
		MPI_Request_free(&req);
	}
	return my_sparse_chunk;
}

int main(int argc, char * argv[]) {
	if (true) {
		/*FILE * pFile = fopen("../sparse1.in", "r");
		 auto tmp = readCSR(pFile);
		 printSparse(tmp);
		 return 0;*/
	}

	int show_results = 0, use_inner = 0, gen_seed = -1, repl_fact = 1, option = -1, exponent = 1, count_ge = 0;
	int num_processes = 1, mpi_rank = 0;
	double comm_start = 0, comm_end = 0, comp_start = 0, comp_end = 0;
	double ge_element = 0;
	SparseMatrix sparse;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	// init MPI
	auto MPI_SPARSE_CELL = commitSparseCellType();

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
					try {
						sparse = readCSR(optarg);
					} catch (runtime_error const& e) {
						fprintf(stderr, "%s : %s", e.what(), optarg);
						MPI_Finalize();
						return 3;
					}
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
	if ((gen_seed == -1) || ((mpi_rank == 0) && sparse.isEmpty())) {
		fprintf(stderr, "error: missing seed or sparse matrix file; exiting\n");
		MPI_Finalize();
		return 3;
	}

	comm_start = MPI_Wtime();
// FIXME: scatter sparse matrix; cache sparse matrix; cache dense matrix
	sparse_type my_sparse_chunk;

	//my_sparse_chunk = sparseScatterV2(sparse, MPI_SPARSE_CELL, num_processes, mpi_rank);
	my_sparse_chunk = sparseIsendIrecv2(sparse, MPI_SPARSE_CELL, num_processes, mpi_rank);
	//my_sparse_chunk = sparseScatterV(sparse, MPI_SPARSE_CELL, num_processes, mpi_rank);
	//	my_sparse_chunk = sparseIsendIrecv(sparse, MPI_SPARSE_CELL, num_processes, mpi_rank);


//	MPI_Gather(&my_sparse, 1, MPI_DOUBLE, sub_avgs, 1, MPI_FLOAT, 0,
//				MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	comm_end = MPI_Wtime();

	if ( debon ) {
//		printf("Send sparse time %d: %.5f\n", mpi_rank, comm_end - comm_start);
		printf("%.5f\n", comm_end - comm_start);
//		deb(my_sparse_chunk.size());
//		debv(my_sparse_chunk);
	}

	//MPI_Type_free(&MPI_SPARSE_CELL );

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

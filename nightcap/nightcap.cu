extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_skein.h"
#include "sph/sph_keccak.h"
#include "sph/sph_cubehash.h"
#include "lyra2/Lyra2.h"
}

#include <miner.h>
#include <cuda_helper.h>

#define WORD_BYTES 4
#define DATASET_BYTES_INIT 536870912
#define DATASET_BYTES_GROWTH 4194304
#define CACHE_BYTES_INIT 8388608
#define CACHE_BYTES_GROWTH 65536
#define EPOCH_LENGTH 60000
#define CACHE_MULTIPLIER 1024
#define MIX_BYTES 64
#define HASH_BYTES 32
#define DATASET_PARENTS 256
#define CACHE_ROUNDS 3
#define ACCESSES 64
#define FNV_PRIME 0x01000193

static uint64_t *d_hash[MAX_GPUS];
static uint64_t* d_matrix[MAX_GPUS];
static char *seed;
static char *cache;
static char *dag;

extern void blake256_cpu_init(int thr_id, uint32_t threads);
extern void blake256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash, int order);
extern void blake256_cpu_setBlock_80(uint32_t *pdata);
extern void keccak256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash, int order);
extern void keccak256_cpu_init(int thr_id, uint32_t threads);
extern void keccak256_cpu_free(int thr_id);
extern void skein256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash, int order);
extern void skein256_cpu_init(int thr_id, uint32_t threads);
extern void cubehash256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash, int order);

extern void lyra2v2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash, int order);
extern void lyra2v2_cpu_init(int thr_id, uint32_t threads, uint64_t* d_matrix);

extern void bmw256_setTarget(const void *ptarget);
extern void bmw256_cpu_init(int thr_id, uint32_t threads);
extern void bmw256_cpu_free(int thr_id);
extern void bmw256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *resultnonces);
int is_prime(uint64_t number) {
	if (number <= 1) return false;
	if((number % 2 == 0) && number > 2) return false;
	for(uint64_t i = 3; i < number / 2; i += 2) {
		if(number % i == 0)
			return true;
	}
	return true;
}

uint32_t fnv(uint32_t v1, uint32_t v2) {
	return ((v1 * FNV_PRIME)  ^ v2) % (0xffffffff);
}

inline uint64_t get_cache_size(uint64_t block_number) {
	uint64_t sz = CACHE_BYTES_INIT + (CACHE_BYTES_GROWTH * floor(((float)block_number / (float)EPOCH_LENGTH)));
	sz -= HASH_BYTES;
	while (!is_prime(sz / HASH_BYTES)) {
		sz -= 2 * HASH_BYTES;
	}
	return sz;
}

inline uint64_t get_full_size(uint64_t block_number) {
	uint64_t sz = DATASET_BYTES_INIT + (DATASET_BYTES_GROWTH * floor((float)block_number / (float)EPOCH_LENGTH));
	sz -= MIX_BYTES;
	while (!is_prime(sz / MIX_BYTES)) {
		sz -= 2 * MIX_BYTES;
	}
	return sz;
}
void nightcap_hash(void *state, const void *input)
{
	uint32_t hashA[8], hashB[8];

	sph_blake256_context      ctx_blake;
	sph_keccak256_context     ctx_keccak;
	sph_skein256_context      ctx_skein;
	sph_bmw256_context        ctx_bmw;
	sph_cubehash256_context   ctx_cube;
	sph_blake256_set_rounds(14);

	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, input, 80);
	sph_blake256_close(&ctx_blake, hashA);

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256(&ctx_keccak, hashA, 32);
	sph_keccak256_close(&ctx_keccak, hashB);

	sph_cubehash256_init(&ctx_cube);
	sph_cubehash256(&ctx_cube, hashB, 32);
	sph_cubehash256_close(&ctx_cube, hashA);

	LYRA2(hashB, 32, hashA, 32, hashA, 32, 1, 4, 4);

	sph_skein256_init(&ctx_skein);
	sph_skein256(&ctx_skein, hashB, 32);
	sph_skein256_close(&ctx_skein, hashA);

	sph_cubehash256_init(&ctx_cube);
	sph_cubehash256(&ctx_cube, hashA, 32);
	sph_cubehash256_close(&ctx_cube, hashB);

	sph_bmw256_init(&ctx_bmw);
	sph_bmw256(&ctx_bmw, hashB, 32);
	sph_bmw256_close(&ctx_bmw, hashA);

	memcpy(state, hashA, 32);
}

void nightcap_hash_48(void *state, const void *input)
{
	uint32_t hashA[8], hashB[8];

	sph_blake256_context      ctx_blake;
	sph_keccak256_context     ctx_keccak;
	sph_skein256_context      ctx_skein;
	sph_bmw256_context        ctx_bmw;
	sph_cubehash256_context   ctx_cube;

	sph_blake256_set_rounds(14);

	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, input, 80);
	sph_blake256_close(&ctx_blake, hashA);

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256(&ctx_keccak, hashA, 48);
	sph_keccak256_close(&ctx_keccak, hashB);

	sph_cubehash256_init(&ctx_cube);
	sph_cubehash256(&ctx_cube, hashB, 48);
	sph_cubehash256_close(&ctx_cube, hashA);

	LYRA2(hashB, 48, hashA, 48, hashA, 48, 1, 4, 4); 

	sph_skein256_init(&ctx_skein);
	sph_skein256(&ctx_skein, hashB, 48);
	sph_skein256_close(&ctx_skein, hashA);

	sph_cubehash256_init(&ctx_cube);
	sph_cubehash256(&ctx_cube, hashA, 48);
	sph_cubehash256_close(&ctx_cube, hashB);

	sph_bmw256_init(&ctx_bmw);
	sph_bmw256(&ctx_bmw, hashB, 48);
	sph_bmw256_close(&ctx_bmw, hashA);

	memcpy(state, hashA, 48);
}

static bool init[MAX_GPUS] = { 0 };

char* mkcache(uint64_t size, char* seed) {
	uint64_t n = floor(size / HASH_BYTES);
	char *cache = (char*)malloc(size);
	nightcap_hash(cache, seed);
	for(int i = 0; i < (n - 1); i++) {
		nightcap_hash(cache + ((i+1)*32), (cache + (i*32)));
	}
	gpulog(LOG_INFO, 1, "Cache initial created.");
	for(int z = 0; z < 3; z++) {
		for(int i = 0; i < n; i++) {
			uint8_t mapped[32];
			uint64_t v = cache[(i*32)] % n;
			for(int k = 0; k < 32; k++) {
				mapped[k] = cache[(((i-1+n) % n)*32)+k] ^ cache[(v*32)+k];
			}
			nightcap_hash(mapped, mapped);
			memcpy(cache + (i*32), mapped, 32);
		}
	}
	return cache;
}
struct dag_work {
	char *cache; 
	uint64_t cache_size;
	//uint64_t i;
	char* dataset;
	uint64_t numnodes;
	uint32_t thread;
};
static void *calc_dataset_item(void *rawk) {
	dag_work *wk = (dag_work*)rawk;
	char *cache = wk->cache; 
	uint64_t cache_size = wk->cache_size;
	//uint64_t i = wk->i;
	char* dataset = wk->dataset;
	uint64_t r = floor(HASH_BYTES / (float)WORD_BYTES);
	uint64_t start = wk->thread * (wk->numnodes/6);
	uint64_t end = (wk->thread + 1) * (wk->numnodes/6);
	/*if(i > r) {
		gpulog(LOG_INFO, 0, "%u", (i*32 % cache_size));
	}*/
	//char *mix = (char*)malloc(32);
	for(uint64_t i = start; i < end; i++) {
	memcpy(dataset + ((i)*32), cache + ((i)*32 % cache_size), 32);
	(dataset + ((i)*32))[0] ^= (i);
		nightcap_hash(dataset + ((i)*32), dataset + ((i)*32));
		for(int j = 0; j < DATASET_PARENTS; j++) {
			uint64_t cache_index = fnv((i) ^ j, (dataset + (i*32))[j % r])*32;
			for(int k = 0; k < 32; k++) {
				(dataset + ((i)*32))[k] = fnv((dataset + ((i)*32))[k], cache[(cache_index % cache_size)+k]);
			}
		}
		nightcap_hash((dataset + ((i)*32)), (dataset + ((i)*32)));
	}
	free(wk);
}

char *calc_dataset(uint64_t full_size, char* cache, uint64_t cache_size) {
	char *dataset = (char*)malloc(full_size);
	/*for(uint64_t i = 0; i < floor(full_size / (float)HASH_BYTES); i += 16) {
	calc_dataset_item<<<1, 16>>>(cache, cache_size, i, dataset);
	}*/
	pthread_t t[6];
	for(int s = 0; s < 6; s++) {
		struct dag_work *wk = (dag_work*)malloc(sizeof(dag_work));
		wk->cache = cache;
		wk->cache_size = cache_size;
		wk->thread = s;
		wk->numnodes = floor(full_size / (float)HASH_BYTES);
		wk->dataset = dataset;
		pthread_create(&t[s], NULL, calc_dataset_item, (void *)wk);
	}
	for(int s = 0; s < 6; s++) {
		pthread_join(t[s], NULL);
	}
	//char *item = calc_dataset_item(cache,cache_size,i);
	//memcpy(dataset + (i*32), item, 32);
	//free(item);
	return dataset;
}

extern "C" int scanhash_nightcap(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t height  = work->height;
	uint64_t cache_size = get_cache_size(height);
	uint64_t dag_size = get_full_size(height);
	const uint32_t first_nonce = pdata[19];
	int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] < 500) ? 18 : is_windows() ? 19 : 20;
	if (strstr(device_name[dev_id], "GTX 10")) intensity = 20;
	uint32_t throughput = cuda_default_throughput(dev_id, 1UL << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x000f;

	if (!init[thr_id])
	{
		seed = (char*)malloc(32);
		memset(seed, 0, 32);
		gpulog(LOG_INFO, thr_id, "DAG size : %u, Cache size: %u", dag_size, cache_size);
		size_t matrix_sz = 16 * sizeof(uint64_t) * 4 * 3;
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		blake256_cpu_init(thr_id, throughput);
		keccak256_cpu_init(thr_id,throughput);
		skein256_cpu_init(thr_id, throughput);
		bmw256_cpu_init(thr_id, throughput);

		// SM 3 implentation requires a bit more memory
		if (device_sm[dev_id] < 500 || cuda_arch[dev_id] < 500)
			matrix_sz = 16 * sizeof(uint64_t) * 4 * 4;
			
		CUDA_SAFE_CALL(cudaMalloc(&d_matrix[thr_id], matrix_sz * throughput));
		lyra2v2_cpu_init(thr_id, throughput, d_matrix[thr_id]);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t)32 * throughput));

		api_set_throughput(thr_id, throughput);
		for(int i = 0; i <= floor(height / EPOCH_LENGTH); i++) {
			nightcap_hash(seed, seed);
		}
		gpulog(LOG_INFO, thr_id, "Generating Cache... Please wait.");	
		cache = mkcache(cache_size, seed);
		gpulog(LOG_INFO, thr_id, "Generating DAG... Please wait.");	
		dag = calc_dataset(dag_size, cache, cache_size);
		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	bmw256_setTarget(ptarget);
	uint64_t n = dag_size / HASH_BYTES;
	uint64_t w = floor(MIX_BYTES / (float)WORD_BYTES);
	uint64_t mixhashes = MIX_BYTES / HASH_BYTES;
	do {
		int order = 0;
		uint8_t hseed[32];
		//gpulog(LOG_INFO, thr_id, "1");
		memcpy(hseed, work->data, 28);
		//gpulog(LOG_INFO, thr_id, "2");
		memcpy(hseed + 28, work->nonces, 4);
		//gpulog(LOG_INFO, thr_id, "3");
		nightcap_hash(hseed, hseed);
		uint8_t mix[64];
		//gpulog(LOG_INFO, thr_id, "4");
		memcpy(mix, hseed, 32);
		//gpulog(LOG_INFO, thr_id, "5");
		memcpy(mix + 32, hseed, 32);
		//gpulog(LOG_INFO, thr_id, "6");
		for(int i = 0; i < ACCESSES; i++) {
			//uint64_t p = fnv(i ^ hseed[0], mix[i % w]) % (unsigned int)floor(n / (float)mixhashes) * mixhashes;
			uint64_t p = 0;
			uint8_t newdata[64];
			for(int k = 0; k < mixhashes; k++) {
				//gpulog(LOG_INFO, thr_id, "7");
				memcpy(newdata, dag + (p + k)*32, 32);
			}
			for(int z = 0; z < 64; z++) {
				mix[z] = fnv(mix[z], newdata[z]);
			}	
		}
		uint8_t cmix[16];
		for(int i = 0; i < 64; i += 4) {
			cmix[i] = fnv(fnv(fnv(mix[i], mix[i+1]), mix[i+2]), mix[i+3]);
		}
		uint8_t finalseed[80];
		//gpulog(LOG_INFO, thr_id, "8");
		memcpy(finalseed, hseed, 32);
		//gpulog(LOG_INFO, thr_id, "9");
		memcpy(finalseed + 32, hseed, 32);
		//gpulog(LOG_INFO, thr_id, "10");
		memcpy(finalseed + 64, cmix, 16);
		//gpulog(LOG_INFO, thr_id, "11");
		blake256_cpu_setBlock_80((uint32_t*)finalseed);
		//gpulog(LOG_INFO, thr_id, "12");
		blake256_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		keccak256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		cubehash256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		lyra2v2_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		skein256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		cubehash256_cpu_hash_32(thr_id, throughput,pdata[19], d_hash[thr_id], order++);

		memset(work->nonces, 0, sizeof(work->nonces));
		bmw256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], work->nonces);	
		*hashes_done = pdata[19] - first_nonce + throughput;
		if (work->nonces[0] != 0)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			lyra2v2_hash(vhash, endiandata);
			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				gpulog(LOG_WARNING, thr_id, "nonce good!", work->nonces[0]);
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					lyra2v2_hash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}
		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart && !abort_flag);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_nightcap(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_matrix[thr_id]);

	bmw256_cpu_free(thr_id);
	keccak256_cpu_free(thr_id);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}

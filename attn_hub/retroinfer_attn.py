from flash_attn import flash_attn_with_kvcache


def retroinfer_prefill_attn(query_states, key_states, value_states, causal):

    attn_out = flash_attn_with_kvcache(
        q=query_states, 
        k_cache=key_states, 
        v_cache=value_states,
        causal=causal
    )
    
    return attn_out



def retroinfer_decode_attn(query_states, key_states, value_states, layer_idx, retroinfer_cache, query_states_next=None):
    
    attn_out = retroinfer_cache.compute(
        query_states.contiguous(), layer_idx, 
        query_states_next.contiguous() if query_states_next is not None else None,
    )
    
    return attn_out

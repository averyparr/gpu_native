use std::{collections::HashMap, ffi::CStr, str::FromStr};

use proc_macro::{TokenStream, TokenTree};
use quote::{quote, quote_spanned, ToTokens};
use syn::{
    parse_quote, parse_quote_spanned, punctuated::Punctuated, spanned::Spanned, token::{self, Bracket, Colon, Extern, Pound, Unsafe}, Abi, Attribute, Block, ExprCall, FnArg, Ident, Item, ItemFn, ItemMod, LitStr, Meta, Pat, PatIdent, PatType, ReturnType, Signature, Stmt, TraitBound, Type, TypePath, Visibility
};

static GPU_TYPES: &str = "one of CUDA, HIP, OneAPI";

fn inline_attr(always: bool, with_span: impl Spanned) -> Attribute {
    let mut inner_meta = "inline";
    if always {
        inner_meta = "inline(always)"
    }
    let meta: Meta = syn::parse_str(inner_meta).expect("Can't parse `inner_meta`");
    let span = with_span.span();
    Attribute {
        pound_token: Pound(span),
        style: syn::AttrStyle::Outer,
        bracket_token: Bracket(span),
        meta: meta,
    }
}

struct ProcMacFailure(TokenStream);

impl From<TokenStream> for ProcMacFailure {
    fn from(value: TokenStream) -> Self {
        Self(value)
    }
}

impl From<syn::Error> for ProcMacFailure {
    fn from(value: syn::Error) -> Self {
        Self(value.to_compile_error().into())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TargetType {
    CUDA,
    HIP,
    OneAPI,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
enum KernelDim {
    One,
    Two,
    Three
}

impl TargetType {
    fn to_arch_gate(&self, for_span: impl Spanned) -> Attribute {
        let attr_meta = match self {
            Self::CUDA => syn::parse_str(r#"cfg(target_arch = "nvptx64")"#).expect("should parse"),
            Self::HIP | Self::OneAPI => todo!(),
        };
        Attribute {
            pound_token: Pound(for_span.span()),
            style: syn::AttrStyle::Outer,
            bracket_token: token::Bracket(for_span.span()),
            meta: attr_meta,
        }
    }

    fn to_many_arch_cfg<'a>(arch_types: impl Iterator<Item = &'a Self>, for_span: impl Spanned) -> Attribute {
        let mut any_str = arch_types.fold(String::from("cfg(any("), |mut acc, curr| {
            match curr {
                TargetType::CUDA => acc.push_str(r#"target_arch = "nvptx64","#),
                TargetType::HIP | TargetType::OneAPI => todo!(),
            }
            acc
        });
        any_str.pop();
        any_str.push_str("))");

        let attr_meta = syn::parse_str(&any_str).expect("Should be able to parse, even for multiple architectures.");

        Attribute {
            pound_token: Pound(for_span.span()),
            style: syn::AttrStyle::Outer,
            bracket_token: token::Bracket(for_span.span()),
            meta: attr_meta,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::CUDA => "cuda",
            Self::HIP => "hip",
            Self::OneAPI => "one_api",
        }
    }

    fn get_kernel_obj_tokens(&self, fn_name: &CStr) -> Result<Vec<Stmt>, TokenStream> {
        let mut out = vec![];
        match self {
                Self::CUDA => {
                        out.push(syn::parse(quote!{
                    use crate::cuda_driver_wrapper::{CUDAKernel, CUDAModule};
                }.into()).map_err(|_| {return quote!{compile_error!("Unable to parse the cuda_driver_wrapper inclusion");};})?);
                out.push(syn::parse(quote!{
                    use std::sync::LazyLock;
                }.into()).map_err(|_| {return quote!{compile_error!("Unable to parse the LazyLock inclusion");};})?);
                out.push(syn::parse(quote!{
                    static MODULE: LazyLock<CUDAModule> = LazyLock::new(|| {
                        let fatbin_data = fatbin_data();
                        unsafe { CUDAModule::new_from_fatbin(fatbin_data as *const _ as *mut _) }
                            .unwrap_or_else(|e| panic!("Unable to load module: Driver error '{e:#?}'"))
                    });
                }.into()).map_err(|_| {return quote!{compile_error!("Unable to parse the static cuda `MODULE` definition.");};})?);
                out.push(syn::parse(quote!{
                    static KERNEL: LazyLock<CUDAKernel> = LazyLock::new(|| {
                        MODULE.get_function(#fn_name).unwrap_or_else(|e| {
                            panic!("Unable to load `{:#?}` from module: Driver error '{e:#?}'", #fn_name);
                        })
                    });
                }.into()).map_err(|_| {return quote!{compile_error!("Unable to parse the static cuda `KERNEL` definition as str");};})?);
            }
            TargetType::HIP => todo!("I don't know how AMD structures it."),
            TargetType::OneAPI => todo!("I don't know how Intel structures it."),
        }
        Ok(out)
    }

    fn link_to_kernel_obj(&self) -> Result<(Stmt, ItemFn), TokenStream> {
        match self {
            TargetType::CUDA => {
                let extern_block = syn::parse(quote! {
                    unsafe extern "Rust" {
                        #[link_name = concat!(env!("CARGO_PKG_NAME"), "_FATBIN_CODE", "_7B4EA9D2")]
                        static FATBIN_DATA: &'static [u8];
                    }
                }.into()).map_err(|_| {return quote!{compile_error!("Unable to parse the unsafe extern Rust block for CUDA device code linking.");};})?;
                let retriver_fn = syn::parse(quote! {
                    fn fatbin_data() -> &'static [u8] {
                        let arr = unsafe { FATBIN_DATA };
                        let mut first_8 = [0; 8];
                        first_8
                            .iter_mut()
                            .enumerate()
                            .for_each(|(i, v)| *v = arr[i]);
                        let maybe_magic = u64::from_le_bytes(first_8);
                        assert_eq!(
                            maybe_magic, 0xB0BACAFEB0BACAFE,
                            "Linked file does not begin with magic value!"
                        );
                        &arr[8..]
                    }
                }.into()).map_err(|_| {return quote!{compile_error!("Unable to parse CUDA's `retriver_fn`");};})?;
                Ok((extern_block, retriver_fn))
            },
            TargetType::HIP => todo!("I don't know how AMD structures it."),
            TargetType::OneAPI => todo!("I don't know how Intel structures it."),
        }
    }

    fn get_kernel_launch_sig(&self, inner_fn_types: &Vec<Type>, kernel_dim: KernelDim) -> Result<Signature, TokenStream> {
        
        let impl_launch: TraitBound = syn::parse(quote! {
            Fn(#(&'kernel mut #inner_fn_types,)*) -> &'kernel CUDASyncObject
        }.into()).map_err(|_| {return quote!{compile_error!("Unable to parse `Fn(&mut...)`");};})?;
            
        let grid_type: TypePath = match kernel_dim {
            KernelDim::One => parse_quote!(crate::cuda_safe::GridDim1D),
            KernelDim::Two => parse_quote!(crate::cuda_safe::GridDim2D),
            KernelDim::Three => parse_quote!(crate::cuda_safe::GridDim3D),
        };

        let block_type: TypePath = match kernel_dim {
            KernelDim::One => parse_quote!(crate::cuda_safe::BlockDim1D),
            KernelDim::Two => parse_quote!(crate::cuda_safe::BlockDim2D),
            KernelDim::Three => parse_quote!(crate::cuda_safe::BlockDim3D),
        };
        
        let sig: Signature = syn::parse(quote!{
            fn launch<'kernel>(
                grid_dim: #grid_type,
                block_dim: #block_type,
                shared_mem_bytes: u32,
                stream: crate::cuda_driver_wrapper::CUDAStream,
            ) -> impl #impl_launch
        }.into()).map_err(|_| {return quote!{compile_error!("Unable to parse `fn launch`");};})?;

        Ok(sig)
    }

    fn kernel_launch(&self) -> Result<Vec<Stmt>, TokenStream> {
        let actual_launch: Stmt = syn::parse(quote!{
        unsafe {
            KERNEL
                .launch(
                    grid_dim.into(),
                    block_dim.into(),
                    shared_mem_bytes,
                    &stream,
                    &mut kernel_params,
                )
                .unwrap_or_else(|e| panic!("Kernel launch failed! Driver error '{e:#?}'"))
        };}.into()).map_err(|_| {return quote!{compile_error!("Unable to parse cuda `KERNEL.launch()`");};})?;
        let sync_return: Stmt = syn::parse(quote! {
            return &crate::cuda_driver_wrapper::CUDASyncObject;
        }.into()).map_err(|_| {return quote!{compile_error!("Unable to parse `&CUDASyncObject`");};})?;

        Ok(vec![actual_launch, sync_return])
    }
}

impl FromStr for TargetType {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "CUDA" => Ok(Self::CUDA),
            "HIP" => Ok(Self::HIP),
            "OneAPI" => Ok(Self::OneAPI),
            _ => Err(()),
        }
    }
}

fn get_architectures<'a>(
    target_ids: impl Iterator<Item = &'a Ident>,
) -> Result<Vec<TargetType>, TokenStream> {
    let mut archs = vec![];

    for target in target_ids {
        let target_type = TargetType::from_str(target.to_string().as_str())
            .map_err(|_| target.clone())
            .map_err(|bad_id| {
                quote_spanned! {bad_id.span()=>
                compile_error!(concat!("Invalid GPU target type! Use ", #GPU_TYPES));
                }
            })?;
        archs.push(target_type);

        // TODO Add support as rustc does!
        if target_type != TargetType::CUDA {
            return Err(quote_spanned! {target.span()=>
                compile_error!("OK, maybe I lied -- only CUDA is supported (for now)");
            }
            .into());
        }
    }

    Ok(archs)
}

fn get_num_ffi_components(specified_types: &Punctuated::<ExprCall, syn::Token![;]>) -> Result<HashMap<String, u8>, TokenStream> {
    specified_types.into_iter().map(|ec| {
        let type_id: String = match ec.func.as_ref() {
            syn::Expr::Path(expr_path) => {
                expr_path.to_token_stream().to_string()
            },
            _ => {
                return Err(quote_spanned! {ec.span()=>
                    compile_error!("Must be a simple function-call-like syntax, e.g. `CUDASlice(2)`")
                }.into())
            },
        };
        if ec.args.len() > 1 {
            return Err(quote_spanned! {ec.span()=>
                compile_error!("Function call cannot have more than one (integral literal) argument.")
            }.into());
        }
        let num_ffi_components = match ec.args.first() {
            Some(int_literal) => {
                if let Ok(val) = u8::from_str(&int_literal.to_token_stream().to_string()) {
                    val
                } else {
                    return Err(quote_spanned! {int_literal.span()=>
                        compile_error!("Cannot parse this as an int literal")
                    }.into());
                }
            }
            None => 1,
        };
        Ok((type_id, num_ffi_components))
    }).collect::<Result<HashMap<_, _>, _>>()
}

fn get_inner_fn_args(sig: &Signature) -> impl Iterator<Item=Result<(Ident, Type), TokenStream>> {
    sig.inputs.iter().map(|inp| -> Result<(Ident, Type), TokenStream> {
            let pat_type = match inp {
                FnArg::Receiver(receiver) => {
                    return Err(quote_spanned! {receiver.span()=>
                        compile_error!("Kernels are only allowed as free functions.");
                    }
                    .into_token_stream()
                    .into());
                }
                FnArg::Typed(pat_type) => pat_type,
            };
            let PatType { pat, ty, .. } = pat_type;
    
            let ident = match pat.as_ref() {
                syn::Pat::Ident(PatIdent {
                    attrs: ident_attrs,
                    by_ref,
                    mutability: _mutability,
                    subpat,
                    ident,
                }) => {
                    if let Some(attribute) = ident_attrs.first() {
                        return Err(quote_spanned! {attribute.span()=>
                            compile_error!("Attributes are not supported *on* variable `ident` tokens.");
                        }.into());
                    }
                    if let Some(ref_tok) = by_ref {
                        return Err(quote_spanned! {ref_tok.span()=>
                            compile_error!("Kernels must take arguments directly (i.e. not using `ref`)");
                        }.into());
                    }
                    if let Some(subpat_token) = subpat {
                        return Err(
                            quote_spanned! {subpat_token.0.span().join(subpat_token.1.span()).expect("Should be from same file")=>
                                compile_error!("Kernels do not support subpattern argument binding");
                            }
                            .into(),
                        );
                    }
                    ident
                }
                _ => return Err(quote_spanned! {pat.span()=>
                    compile_error!("Only 'standard' function arguments are accepted, e.g. 'arg: T'");
                }
                .into()),
            };
    
            match ty.as_ref() {
                syn::Type::Array(_) | syn::Type::Path(_) | syn::Type::Ptr(_) => Ok(()),
                syn::Type::BareFn(_) => Err(quote_spanned! {ty.span()=>
                    compile_error!("Closures (and function pointers) are *very* FFI unsafe.");
                }),
                syn::Type::Group(_) => Err(quote_spanned! {ty.span()=>
                    compile_error!("I don't really understand what group types are. Try something simpler?");
                }),
                syn::Type::ImplTrait(_) => Err(quote_spanned! {ty.span()=>
                    // TODO
                    compile_error!("Generic support is not yet present, but will eventually be added.");
                }),
                syn::Type::Infer(_) => Err(quote_spanned! {ty.span()=>
                    compile_error!("Inferred types are not allowed for GPU kernels.");
                }),
                syn::Type::Macro(_) => Err(quote_spanned! {ty.span()=>
                    compile_error!("Macros are not allowed in GPU kernel type creation, because we can't make the resulting code FFI safe.");
                }),
                syn::Type::Never(_) => Err(quote_spanned! {ty.span()=>
                    compile_error!("Passing arguments of type `!` is not supported for GPU kernels.");
                }),
                syn::Type::Paren(type_paren) => Err(quote_spanned! {type_paren.span()=>
                    compile_error!("Please remove the parantheses around the type.");
                }),
                syn::Type::Reference(_) => Err(quote_spanned! {ty.span()=>
                    compile_error!("As far as *I* know, references will not be correctly passed to the GPU. Only owned types can be passed.");
                }),
                syn::Type::Slice(_) => Err(quote_spanned! {ty.span()=>
                    // TODO: this probably needs updates if I generalize from CUDA.
                    compile_error!("Slices are not FFI-safe. Try using `slices::CUDASlice` or something similar.");
                }),
                syn::Type::TraitObject(_) => Err(quote_spanned! {ty.span()=>
                    compile_error!("Trait objects cannot be supported for kernels. A concrete type must be given.");
                }),
                syn::Type::Tuple(_) => Err(quote_spanned! {ty.span()=>
                    compile_error!("Tuples are not currently supported; it's unknown whether they are FFI safe for kernels.");
                }),
                syn::Type::Verbatim(_) => Err(quote_spanned! {ty.span()=>
                    compile_error!("Even `syn` couldn't parse this type. I don't know what to do with that.");
                }),
                _ => todo!(),
            }?;
            Ok((ident.clone(), ty.as_ref().clone()))
        })
}

/// I really just want something in the standard library which performs a transformation
/// impl Iterator<Item = Result<
///     impl Iterator<Item = 
///         Result<T, E>
///     >,
///     E
/// >
/// 
/// to 
/// 
/// impl Iterator<Item = Result<T, E>>
/// 
/// but I couldn't find something that did what I wanted without some calls to .collect().
/// This performs that operation
#[allow(unused)]
struct FlatErrorHolder<T, E, Outer: Iterator<Item = Result<Inner, E>>, Inner: Iterator<Item = Result<T, E>>>(Outer, Option<Inner>);
impl<T, E, Outer: Iterator<Item = Result<Inner, E>>, Inner: Iterator<Item = Result<T, E>>> Iterator for FlatErrorHolder<T, E, Outer, Inner> {
    type Item = Result<T, E>;
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        match &mut self.1 {
            Some(inner_iter) => inner_iter.next(),
            None => {
                match self.0.next() {
                    Some(v) => {
                        match v {
                            Ok(mut inner) => {let ret = inner.next(); self.1 = Some(inner); ret},
                            Err(e) => Some(Err(e))
                        }
                    },
                    None => None
                }
            }
        }
    }
}

fn get_ffi_fn_args(sig: &Signature, ident_to_num_ffi: &HashMap<String, u8>) -> impl Iterator<Item=Result<impl Iterator<Item=Result<(Ident, Type, u8), TokenStream>>, TokenStream>> {
    let multiple_iter = get_inner_fn_args(sig).map(|arg| -> Result<_, TokenStream> {
        let (id, ty) = arg?;
        let ty_str = ty.to_token_stream().to_string();
        let matching_types = ident_to_num_ffi
            .keys()
            .filter(|known_id| 
                ty_str.contains(known_id.as_str())
            ).max_by(|a, b| 
                a.len().cmp(&b.len())
            );
        let num_ffi_parts = if let Some(matching_known_ty) = matching_types {
            *ident_to_num_ffi.get(matching_known_ty).expect("Got key from hashmap.keys()")
        } else {
            return Err(quote_spanned! {ty.span()=>
                compile_error!("Unable to determine the number of FFI components. 
                Consider passing it to `gpu_kernel` like `#[gpu_kernel(CUDA | CUDASliceMut(2))]`");
            }.into());
        };

        let res = (0..num_ffi_parts).map(move |ffi_comp| {
            let sub_id = Ident::new(&format!("{id}_{ffi_comp}"), id.span());
            let sub_type: Type = parse_quote_spanned! {ty.span()=>
                <<#ty as crate::cuda_safe::GPUPassable>::FFIRep as crate::cuda_safe::TupleIndex<#num_ffi_parts, #ffi_comp>>::Ty
            };
            Ok((sub_id, sub_type, ffi_comp))
        });
        Ok(res)
    });
    multiple_iter
}

enum FFIConversion {
    ToFFI,
    FromFFI
}

impl FFIConversion {
    fn perform_conversion(&self, sig: &Signature, inner_fn_args: &Vec<Result<(Ident, Type), TokenStream>>, ident_to_num_ffi: &HashMap<String, u8>) -> Result<(Vec<Stmt>, Vec<(Ident, Type)>), TokenStream> {
        let mut ffi_conversions = vec![];
        let mut all_ffi_ids = vec![];
        
        for (ff_assoc_vars, orig_param) in get_ffi_fn_args(sig, ident_to_num_ffi).zip(inner_fn_args.iter()) {
            let ff_assoc_vars = ff_assoc_vars?.collect::<Result<Vec<_>,_>>()?;
            let ffi_ids = ff_assoc_vars.iter().map(|x| &x.0).collect::<Vec<_>>();
            let ffi_tys = ff_assoc_vars.iter().map(|x| &x.1).collect::<Vec<_>>();
            let (id, ty) = orig_param.as_ref().map_err(|e| e.clone())?;
            let tuple_id = Ident::new(&format!("{id}_tuple"), id.span());

            let id_type_span = id.span().join(ty.span()).expect("Should be in the same file");
            match self {
                Self::ToFFI => {
                    let tuple_typecheck: Stmt = syn::parse(quote_spanned! {id_type_span=>
                        let #tuple_id: (#(#ffi_tys,)*) = #id.to_ffi();
                    }.into()).map_err(|_| {return quote!{compile_error!("Unable to parse tuple typecheck of FFI args");};})?;
                    ffi_conversions.push(tuple_typecheck);
                    let tuple_destructuring: Stmt = syn::parse(quote_spanned! {id_type_span=>
                        let (#(mut #ffi_ids),*) = #tuple_id;
                    }.into()).map_err(|_| {return quote!{compile_error!("Unable to parse destructuring of FFI args");};})?;
                    ffi_conversions.push(tuple_destructuring);
                },
                Self::FromFFI => {
                    let arg_restructuring: Stmt = syn::parse(quote_spanned! {id_type_span=>
                        let #id = unsafe { <#ty as crate::cuda_safe::GPUPassable>::from_ffi((#(#ffi_ids,)*)) };
                    }.into()).map_err(|_| {return quote!{compile_error!("Unable to parse restructuring of FFI args");};})?;
                    ffi_conversions.push(arg_restructuring);
                }
            }
            all_ffi_ids.extend(ff_assoc_vars.into_iter().map(|(id, ty, _)| (id, ty)));
        }

        Ok((ffi_conversions, all_ffi_ids))
    }
}

fn generate_outer_function(
    vis: &Visibility,
    sig: &Signature,
    target_type: TargetType,
    ident_to_num_ffi: &HashMap<String, u8>
) -> Result<ItemFn, TokenStream> {
    let outer_fn_name = format!("{}_kernel", sig.ident.to_string());

    let const_problem = sig.constness.map(|c| {
        return quote_spanned! {c.span()=>
            compile_error!("`const` kernels are not allowed!");
        };
    });
    let async_problem = sig.asyncness.map(|a| {
        return quote_spanned! {a.span()=>
            compile_error!("(Rust language feature) `async` kernels are not allowed");
        };
    });
    let abi_problem = sig.abi.as_ref().map(|a| {
        return quote_spanned! {a.span()=>
            compile_error!("Non-native-ABIs are not currently supported (though this will be transformed into an extern 'ptx-kernel' internally");
    };
    });
    let generic_problem = sig.generics.params.first().map(|_| {
        return quote_spanned! {sig.generics.span()=>
            // TODO
            compile_error!("Generic support is not yet present, but will eventually be added.");
        };
    });

    if let Some(problem) = const_problem
        .into_iter()
        .chain(async_problem.into_iter())
        .chain(abi_problem.into_iter())
        .chain(generic_problem.into_iter())
        .next()
    {
        return Err(problem.into_token_stream().into());
    };

    let orig_inputs = &sig.inputs;
    let mut input_idents = vec![];
    
    let mut kernel_inputs = Punctuated::new();

    let (restructure_stmts, all_ffi_ids) = FFIConversion::FromFFI.perform_conversion(sig, &get_inner_fn_args(sig).collect(), ident_to_num_ffi)?;
    for (ffi_id, ffi_ty) in all_ffi_ids {
        let pat = Pat::Ident(PatIdent { attrs: vec![], by_ref: None, mutability: None, ident: ffi_id, subpat: None });
        let to_push = FnArg::Typed(PatType { attrs: vec![], pat: Box::new(pat), colon_token: Colon(ffi_ty.span()), ty: Box::new(ffi_ty) });
        kernel_inputs.push(to_push);
    }

    for inp in orig_inputs {
        let pat_type = match inp {
            FnArg::Receiver(receiver) => {
                return Err(quote_spanned! {receiver.span()=>
                    compile_error!("Kernels are only allowed as free functions.");
                }
                .into_token_stream()
                .into());
            }
            FnArg::Typed(pat_type) => pat_type,
        };
        let PatType { pat, ty, .. } = pat_type;
        let pat_ident;
        match pat.as_ref() {
            syn::Pat::Ident(PatIdent {
                attrs: ident_attrs,
                by_ref,
                mutability: _,
                subpat,
                ident: ident_to_clone,
            }) => {
                pat_ident = ident_to_clone.clone();
                if let Some(attribute) = ident_attrs.first() {
                    return Err(quote_spanned! {attribute.span()=>
                        compile_error!("Attributes are not supported *on* variable `ident` tokens.");
                    }.into());
                }
                if let Some(ref_tok) = by_ref {
                    return Err(quote_spanned! {ref_tok.span()=>
                        compile_error!("Kernels must take arguments directly (i.e. not using `ref`)");
                    }.into());
                }
                if let Some(subpat_token) = subpat {
                    return Err(
                        quote_spanned! {subpat_token.0.span().join(subpat_token.1.span()).expect("Should be from same file")=>
                            compile_error!("Kernels do not support subpattern argument binding");
                        }
                        .into(),
                    );
                }
            }
            _ => return Err(quote_spanned! {pat.span()=>
                compile_error!("Only 'standard' function arguments are accepted, e.g. 'arg: T'");
            }
            .into()),
        }

        input_idents.push(pat_ident.clone());

        let ty: &syn::Type = ty.as_ref();
        match ty {
            syn::Type::Array(_) | syn::Type::Path(_) | syn::Type::Ptr(_) => Ok(()),
            syn::Type::BareFn(_) => Err(quote_spanned! {ty.span()=>
                compile_error!("Closures (and function pointers) are *very* FFI unsafe.");
            }),
            syn::Type::Group(_) => Err(quote_spanned! {ty.span()=>
                compile_error!("I don't really understand what group types are. Try something simpler?");
            }),
            syn::Type::ImplTrait(_) => Err(quote_spanned! {ty.span()=>
                // TODO
                compile_error!("Generic support is not yet present, but will eventually be added.");
            }),
            syn::Type::Infer(_) => Err(quote_spanned! {ty.span()=>
                compile_error!("Inferred types are not allowed for GPU kernels.");
            }),
            syn::Type::Macro(_) => Err(quote_spanned! {ty.span()=>
                compile_error!("Macros are not allowed in GPU kernel type creation, because we can't make the resulting code FFI safe.");
            }),
            syn::Type::Never(_) => Err(quote_spanned! {ty.span()=>
                compile_error!("Passing arguments of type `!` is not supported for GPU kernels.");
            }),
            syn::Type::Paren(type_paren) => Err(quote_spanned! {type_paren.span()=>
                compile_error!("Please remove the parantheses around the type.");
            }),
            syn::Type::Reference(_) => Err(quote_spanned! {ty.span()=>
                compile_error!("As far as *I* know, references will not be correctly passed to the GPU. Only owned types can be passed.");
            }),
            syn::Type::Slice(_) => Err(quote_spanned! {ty.span()=>
                // TODO: this probably needs updates if I generalize from CUDA.
                compile_error!("Slices are not FFI-safe. Try using `slices::CUDASlice` or something similar.");
            }),
            syn::Type::TraitObject(_) => Err(quote_spanned! {ty.span()=>
                compile_error!("Trait objects cannot be supported for kernels. A concrete type must be given.");
            }),
            syn::Type::Tuple(_) => Err(quote_spanned! {ty.span()=>
                compile_error!("Tuples are not currently supported; it's unknown whether they are FFI safe for kernels.");
            }),
            syn::Type::Verbatim(_) => Err(quote_spanned! {ty.span()=>
                compile_error!("Even `syn` couldn't parse this type. I don't know what to do with that.");
            }),
            _ => todo!(),
        }?;
    }

    assert!(kernel_inputs.len() == 8);

    let kernel_sig = Signature {
        constness: None,
        asyncness: None,
        unsafety: Some(Unsafe(
            sig.unsafety
                .map(|u| u.span())
                .unwrap_or(sig.fn_token.span()),
        )),
        abi: Some(Abi {
            extern_token: Extern(sig.span()),
            name: Some(LitStr::new("ptx-kernel", sig.span())),
        }),
        fn_token: sig.fn_token,
        ident: Ident::new(&outer_fn_name, sig.ident.span()),
        generics: sig.generics.clone(),
        paren_token: sig.paren_token,
        inputs: kernel_inputs,
        variadic: None,
        output: ReturnType::Default,
    };

    let attrs = vec![target_type.to_arch_gate(kernel_sig.span())];
    let mut block = {
        let b: Block = syn::parse_quote_spanned! {sig.span()=>{}};
        Box::new(b)
    };

    block.stmts.extend(restructure_stmts);

    let inner_fn_name = format!("{}_inner", sig.ident.to_string());
    let mut inner_fn_name: Ident =
        syn::parse_str(inner_fn_name.as_str()).map_err(|_| {return quote!{compile_error!("Unable to parse `inner_fn_name` as str");};})?;
    inner_fn_name.set_span(sig.ident.span());
    let fn_call = parse_quote_spanned! {sig.span()=>
        #inner_fn_name(#(#input_idents,)*);
    };
    block.stmts.push(fn_call);

    Ok(ItemFn {
        attrs: attrs,
        vis: vis.clone(),
        sig: kernel_sig,
        block,
    })
}

fn make_gpu_mod(input_fn: &ItemFn, arch: TargetType, ident_to_num_ffi: &HashMap<String, u8>) -> Result<ItemMod, ProcMacFailure> {
    let ItemFn { vis, sig, .. } = input_fn;

    let kernel_name = format!("{}_{}_kernel", sig.ident.to_string(), arch.as_str());
    let mut gpu_mod: ItemMod = syn::parse_str(&format!("mod {} {{}}", kernel_name))?;
    gpu_mod.vis = vis.clone();
    gpu_mod.unsafety = sig.unsafety;
    gpu_mod.attrs.push(arch.to_arch_gate(sig.span()));

    let inner_fn_name = format!("{}_inner", sig.ident.to_string());

    let items = &mut gpu_mod.content.as_mut().unwrap().1;
    let mut inner_fn = input_fn.clone();
    inner_fn.sig.ident = Ident::new(&inner_fn_name, inner_fn.sig.span());

    inner_fn.attrs.push(inline_attr(true, inner_fn.sig.span()));

    items.push(Item::Use(syn::parse_str("use super::*;")?));
    items.push(Item::Use(syn::parse_str(
        "use crate::cuda_safe::GPUPassable;",
    )?));
    items.push(Item::Fn(inner_fn));

    items.push(Item::Fn(generate_outer_function(vis, sig, arch, ident_to_num_ffi)?));

    Ok(gpu_mod)
}

fn make_cpu_mod(input_fn: &ItemFn, arch: TargetType, ident_to_num_ffi: &HashMap<String, u8>, kernel_dim: KernelDim) -> Result<ItemMod, ProcMacFailure> {
    let ItemFn { vis, sig, .. } = input_fn;
    let fn_ident = &sig.ident;

    let (static_linker, getter) = arch.link_to_kernel_obj()?;

    let mut mod_structure: ItemMod = parse_quote! {

        #[cfg(not(any(target_arch = "nvptx64")))]
        #vis mod #fn_ident {
            use super::*;
            use crate::cuda_safe::GPUPassable;
            use crate::cuda_driver_wrapper::CUDASyncObject;

            #static_linker
            #getter
        }
    };

    let inner_fn_args: Vec<_> = get_inner_fn_args(sig).collect();
    let inner_fn_types = inner_fn_args.iter().map(|e| e.as_ref().map(|x| x.1.clone()).map_err(|e| e.clone())).collect::<Result<Vec<_>,_>>()?;
    let inner_fn_idents = inner_fn_args.iter().map(|e| e.as_ref().map(|x| x.0.clone()).map_err(|e| e.clone())).collect::<Result<Vec<_>, _>>()?;

    let kernel_launch_sig = arch.get_kernel_launch_sig(&inner_fn_types, kernel_dim)?;
    
    let (ffi_conversions, all_ffi_ids) = FFIConversion::ToFFI.perform_conversion(sig, &inner_fn_args, ident_to_num_ffi)?;
    let all_ffi_ids = all_ffi_ids.into_iter().map(|(id, _)| id);
    
    let fn_name_in_module = std::ffi::CString::from_str(&format!("{}_kernel", sig.ident.to_string())).expect("Should be no interior null bytes.");

    let content = &mut mod_structure
        .content
        .as_mut()
        .expect("Should have nonempty mod")
        .1;
    
    let bin_data = arch.get_kernel_obj_tokens(&fn_name_in_module)?;
    let kernel_launch = arch.kernel_launch()?;

    content.push(syn::parse(quote_spanned! {sig.span()=>
        #[cfg(not(any(target_arch = "nvptx64")))]
        pub #kernel_launch_sig {
            #(#bin_data)*

            move |#(#inner_fn_idents,)*|{
                #(#ffi_conversions)*

                let mut kernel_params = [
                    #(&mut #all_ffi_ids as *mut _ as *mut core::ffi::c_void,)*
                ];

                #(#kernel_launch)*
            }
        }
    }.into())?);

    Ok(mod_structure)
}

fn gpu_kernel_inner(attr: TokenStream, item: TokenStream) -> Result<TokenStream, ProcMacFailure> {
    let input_fn: ItemFn = syn::parse(item)?;

    let mut first_stmt = TokenStream::new();
    let mut second_stmt = TokenStream::new();
    let mut add_to_first = true;
    for v in attr.into_iter() {
        if let TokenTree::Punct(p) = &v {
            if *p == '|' {
                add_to_first = false;
                continue;
            }
        }
        if add_to_first {
            first_stmt.extend(TokenStream::from(v));
        } else {
            second_stmt.extend(TokenStream::from(v));
        }
    }


    let arch_kernel_types = syn::parse::Parser::parse(
        Punctuated::<Ident, syn::Token![,]>::parse_terminated,
        first_stmt,
    )?;

    let arch_types: Vec<_> = arch_kernel_types.iter().filter(|&id| {
        !(id == "1D" || id == "2D" || id == "3D")
    }).collect();

    let kernel_dim_args: Vec<_> = arch_kernel_types.iter().filter(|&id| {
        id == "1D" || id == "2D" || id == "3D"
    }).collect();

    if kernel_dim_args.len() > 1 {
        return Err(ProcMacFailure(quote_spanned! {kernel_dim_args[1].span()=>
            compile_error!("Cannot pass more than one kernel dimension!");
        }.into()));
    }

    let kernel_dim = kernel_dim_args.first().map(|id| match id.to_string().as_str() {
        "1D" => KernelDim::One,
        "2D" => KernelDim::Two,
        "3D" => KernelDim::Three,
        _ => unreachable!("Should have filtered it out."),
    }).unwrap_or(KernelDim::Three);

    let arch_compile = get_architectures(arch_types.into_iter())?;
    if arch_compile.len() == 0 {
        return Err(ProcMacFailure(quote! {target_ids.span()=>
            compile_error!(concat!("Must provide one or more GPU types (right now, ", #GPU_TYPES, ")."));
        }
        .into()));
    }
    let ffi_component_info = syn::parse::Parser::parse(
        Punctuated::<ExprCall, syn::Token![;]>::parse_terminated,
        second_stmt
    )?;
    let ident_to_num_ffi = get_num_ffi_components(&ffi_component_info)?;

    // We do this because we want a proc_macro2::TokenStream, not the
    // proc_macro::TokenStream which is visible in the crate.
    let mut output_token_stream = quote! {};

    for arch in arch_compile {
        output_token_stream.extend(make_gpu_mod(&input_fn, arch, &ident_to_num_ffi)?.to_token_stream());
        output_token_stream.extend(make_cpu_mod(&input_fn, arch, &ident_to_num_ffi, kernel_dim)?.to_token_stream());
    }

    Ok(output_token_stream.into())
}

#[proc_macro_attribute]
pub fn gpu_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    match gpu_kernel_inner(attr, item) {
        Ok(v) | Err(ProcMacFailure(v)) => v,
    }
}

#[proc_macro_attribute]
pub fn host(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item = syn::parse_macro_input!(item as Item);
    
    let output = quote! {
        #[cfg(not(any(target_arch = "nvptx64")))]
        #item
    };
    
    output.into()
}

#[proc_macro_attribute]
pub fn device(attr: TokenStream, item: TokenStream) -> TokenStream {
    let item = syn::parse_macro_input!(item as Item);

    let maybe_arch_types_ids = syn::parse::Parser::parse(
        Punctuated::<Ident, syn::Token![,]>::parse_terminated,
        attr,
    );
    let arch_types_ids = match maybe_arch_types_ids {
        Ok(v) => v,
        Err(_) => {
            return quote!{
                compile_error!(concat!("`device` take either no arguments or", #GPU_TYPES));
            }.into();
        }
    };
    let maybe_archs  = get_architectures(arch_types_ids.iter());
    let arch_types = match maybe_archs {
        Ok(v) => v,
        Err(e) => {
            return e;
        }
    };

    if arch_types.is_empty() {
        return quote! {
            #[cfg(any(target_arch = "nvptx64"))]
            #item
        }.into();
    } else {
        let attr = TargetType::to_many_arch_cfg(arch_types.iter(), arch_types_ids.span());
        return quote! {
            #attr
            #item
        }.into();
    };
}
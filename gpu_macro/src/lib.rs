use std::str::FromStr;

use proc_macro::TokenStream;
use quote::{ToTokens, quote, quote_spanned};
use syn::{
    Attribute, Ident, Item, ItemFn, ItemMod, Meta, Visibility,
    parse::{self, Parse},
    parse_macro_input,
    punctuated::Punctuated,
    spanned::Spanned,
    token::{self, Brace, Bracket, Comma, Pound},
};

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

impl TargetType {
    fn to_arch_gate(&self, for_span: impl Spanned) -> Result<Attribute, syn::Error> {
        let attr_meta = match self {
            Self::CUDA => syn::parse_str(r#"cfg(target_arch = "nvptx64")"#)?,
            Self::HIP | Self::OneAPI => todo!(),
        };
        Ok(Attribute {
            pound_token: Pound(for_span.span()),
            style: syn::AttrStyle::Outer,
            bracket_token: token::Bracket(for_span.span()),
            meta: attr_meta,
        })
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::CUDA => "cuda",
            Self::HIP => "hip",
            Self::OneAPI => "one_api",
        }
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

fn get_architectures(
    target_ids: &Punctuated<Ident, Comma>,
) -> Result<Vec<TargetType>, TokenStream> {
    let mut archs = vec![];

    let gpu_types = "one of CUDA, HIP, OneAPI";

    for target in target_ids {
        let target_type = TargetType::from_str(target.to_string().as_str())
            .map_err(|_| target.clone())
            .map_err(|bad_id| {
                quote_spanned! {bad_id.span()=>
                compile_error!(concat!("Invalid GPU target type! Use ", #gpu_types));
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

    if archs.len() == 0 {
        return Err(quote! {target_ids.span()=>
            compile_error!(concat!("Must provide one or more GPU types (right now, ", #gpu_types, ")."));
        }
        .into());
    }

    Ok(archs)
}

fn gpu_kernel_inner(attr: TokenStream, item: TokenStream) -> Result<TokenStream, ProcMacFailure> {
    let mut input_fn: ItemFn = syn::parse(item)?;

    let ItemFn {
        attrs,
        vis,
        sig,
        block,
    } = &input_fn;

    let arch_types =
        syn::parse::Parser::parse(Punctuated::<Ident, syn::Token![,]>::parse_terminated, attr)?;

    let arch_compile = get_architectures(&arch_types)?;

    // We do this because we want a proc_macro2::TokenStream, not the
    // proc_macro::TokenStream which is visible in the crate.
    let mut output_token_stream = quote! {};

    for arch in arch_compile {
        let kernel_name = format!("{}_{}_kernel", sig.ident.to_string(), arch.as_str());
        let mut gpu_mod: ItemMod = syn::parse_str(&format!("mod {} {{}}", kernel_name))?;
        gpu_mod.vis = vis.clone();
        gpu_mod.unsafety = sig.unsafety;
        gpu_mod.attrs.push(arch.to_arch_gate(sig.span())?);

        let inner_fn_name = format!("{}_inner", sig.ident.to_string());
        let outer_fn_name = format!("{}_kernel", sig.ident.to_string());

        let items = &mut gpu_mod.content.as_mut().unwrap().1;
        let mut inner_fn = input_fn.clone();
        inner_fn.sig.ident = Ident::new(&inner_fn_name, inner_fn.sig.span());

        inner_fn.attrs.push(inline_attr(true, inner_fn.sig.span()));

        items.push(Item::Fn(inner_fn));

        let v = gpu_mod.to_token_stream();
        output_token_stream.extend(v);
    }

    if !matches!(vis, Visibility::Public(_)) {
        quote! {
            compile_error!("Function must be public!")
        };
    }

    let sig_span = sig.span();

    let output = input_fn.to_token_stream();

    let mut tokens = quote_spanned! {sig_span=>
        #[cfg(target_arch = "nvptx64")]
    };

    Ok(output_token_stream.into())
}

#[proc_macro_attribute]
pub fn gpu_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    match gpu_kernel_inner(attr, item) {
        Ok(v) | Err(ProcMacFailure(v)) => v,
    }
}

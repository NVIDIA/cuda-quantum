/* Copyright (c) 2007-2014 Massachusetts Institute of Technology
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "nlopt-internal.h"

/*********************************************************************/

#ifdef NLOPT_CXX
#include "stogo.h"
#endif

#include "cdirect.h"
#include "mma.h"
#include "cobyla.h"
#include "neldermead.h"

/*********************************************************************/
/* wrapper functions, only for derivative-free methods, that
   eliminate dimensions with lb == ub.   (The gradient-based methods
   should handle this case directly, since they operate on much
   larger vectors where I am loathe to make copies unnecessarily.) */

typedef struct {
    nlopt_func f;
    nlopt_mfunc mf;
    void *f_data;
    unsigned n;                 /* true dimension */
    double *x;                  /* scratch vector of length n */
    double *grad;               /* optional scratch vector of length n */
    const double *lb, *ub;      /* bounds, of length n */
} elimdim_data;

static void *elimdim_makedata(nlopt_func f, nlopt_mfunc mf, void *f_data, unsigned n, double *x, const double *lb, const double *ub, double *grad)
{
    elimdim_data *d = (elimdim_data *) malloc(sizeof(elimdim_data));
    if (!d)
        return NULL;
    d->f = f;
    d->mf = mf;
    d->f_data = f_data;
    d->n = n;
    d->x = x;
    d->lb = lb;
    d->ub = ub;
    d->grad = grad;
    return d;
}

static double elimdim_func(unsigned n0, const double *x0, double *grad, void *d_)
{
    elimdim_data *d = (elimdim_data *) d_;
    double *x = d->x;
    const double *lb = d->lb, *ub = d->ub;
    double val;
    unsigned n = d->n, i, j;

    (void) n0;                  /* unused */
    for (i = j = 0; i < n; ++i) {
        if (lb[i] == ub[i])
            x[i] = lb[i];
        else                    /* assert: j < n0 */
            x[i] = x0[j++];
    }
    val = d->f(n, x, grad ? d->grad : NULL, d->f_data);
    if (grad) {
        /* assert: d->grad != NULL */
        for (i = j = 0; i < n; ++i)
            if (lb[i] != ub[i])
                grad[j++] = d->grad[i];
    }
    return val;
}

static void elimdim_mfunc(unsigned m, double *result, unsigned n0, const double *x0, double *grad, void *d_)
{
    elimdim_data *d = (elimdim_data *) d_;
    double *x = d->x;
    const double *lb = d->lb, *ub = d->ub;
    unsigned n = d->n, i, j;

    (void) n0;                  /* unused */
    (void) grad;                /* assert: grad == NULL */
    for (i = j = 0; i < n; ++i) {
        if (lb[i] == ub[i])
            x[i] = lb[i];
        else                    /* assert: j < n0 */
            x[i] = x0[j++];
    }
    d->mf(m, result, n, x, NULL, d->f_data);
}

/* compute the eliminated dimension: number of dims with lb[i] != ub[i] */
static unsigned elimdim_dimension(unsigned n, const double *lb, const double *ub)
{
    unsigned n0 = 0, i;
    for (i = 0; i < n; ++i)
        n0 += lb[i] != ub[i] ? 1U : 0;
    return n0;
}

/* modify v to "shrunk" version, with dimensions for lb[i] == ub[i] elim'ed */
static void elimdim_shrink(unsigned n, double *v, const double *lb, const double *ub)
{
    unsigned i, j;
    if (v)
        for (i = j = 0; i < n; ++i)
            if (lb[i] != ub[i])
                v[j++] = v[i];
}

/* inverse of elimdim_shrink */
static void elimdim_expand(unsigned n, double *v, const double *lb, const double *ub)
{
    unsigned i, j;
    if (v && n > 0) {
        j = elimdim_dimension(n, lb, ub) - 1;
        for (i = n - 1; i > 0; --i) {
            if (lb[i] != ub[i])
                v[i] = v[j--];
            else
                v[i] = lb[i];
        }
        if (lb[0] == ub[0])
            v[0] = lb[0];
    }
}

/* given opt, create a new opt with equal-constraint dimensions eliminated */
static nlopt_opt elimdim_create(nlopt_opt opt)
{
    nlopt_opt opt0;
    nlopt_munge munge_copy_save = opt->munge_on_copy;
    double *x, *grad = NULL;
    unsigned i;

    opt->munge_on_copy = 0;     /* hack: since this is an internal copy,
                                   we can leave it un-munged; see issue #26 */
    opt0 = nlopt_copy(opt);
    opt->munge_on_copy = munge_copy_save;
    if (!opt0)
        return NULL;
    x = (double *) malloc(sizeof(double) * opt->n);
    if (opt->n && !x) {
        nlopt_destroy(opt0);
        return NULL;
    }

    if (opt->algorithm == NLOPT_GD_STOGO || opt->algorithm == NLOPT_GD_STOGO_RAND) {
        grad = (double *) malloc(sizeof(double) * opt->n);
        if (opt->n && !grad)
            goto bad;
    }

    opt0->n = elimdim_dimension(opt->n, opt->lb, opt->ub);
    elimdim_shrink(opt->n, opt0->lb, opt->lb, opt->ub);
    elimdim_shrink(opt->n, opt0->ub, opt->lb, opt->ub);
    elimdim_shrink(opt->n, opt0->xtol_abs, opt->lb, opt->ub);
    elimdim_shrink(opt->n, opt0->dx, opt->lb, opt->ub);

    opt0->munge_on_destroy = opt0->munge_on_copy = NULL;

    opt0->f = elimdim_func;
    opt0->f_data = elimdim_makedata(opt->f, NULL, opt->f_data, opt->n, x, opt->lb, opt->ub, grad);
    if (!opt0->f_data)
        goto bad;

    for (i = 0; i < opt->m; ++i) {
        opt0->fc[i].f = opt0->fc[i].f ? elimdim_func : NULL;
        opt0->fc[i].mf = opt0->fc[i].mf ? elimdim_mfunc : NULL;
        opt0->fc[i].f_data = elimdim_makedata(opt->fc[i].f, opt->fc[i].mf, opt->fc[i].f_data, opt->n, x, opt->lb, opt->ub, NULL);
        if (!opt0->fc[i].f_data)
            goto bad;
    }

    for (i = 0; i < opt->p; ++i) {
        opt0->h[i].f = opt0->h[i].f ? elimdim_func : NULL;
        opt0->h[i].mf = opt0->h[i].mf ? elimdim_mfunc : NULL;
        opt0->h[i].f_data = elimdim_makedata(opt->h[i].f, opt->h[i].mf, opt->h[i].f_data, opt->n, x, opt->lb, opt->ub, NULL);
        if (!opt0->h[i].f_data)
            goto bad;
    }

    return opt0;
  bad:
    free(grad);
    free(x);
    nlopt_destroy(opt0);
    return NULL;
}

/* like nlopt_destroy, but also frees elimdim_data */
static void elimdim_destroy(nlopt_opt opt)
{
    unsigned i;
    if (!opt)
        return;

    free(((elimdim_data *) opt->f_data)->x);
    free(((elimdim_data *) opt->f_data)->grad);
    free(opt->f_data);
    opt->f_data = NULL;

    for (i = 0; i < opt->m; ++i) {
        free(opt->fc[i].f_data);
        opt->fc[i].f_data = NULL;
    }
    for (i = 0; i < opt->p; ++i) {
        free(opt->h[i].f_data);
        opt->h[i].f_data = NULL;
    }

    nlopt_destroy(opt);
}

/* return whether to use elimdim wrapping. */
static int elimdim_wrapcheck(nlopt_opt opt)
{
    if (!opt)
        return 0;
    if (elimdim_dimension(opt->n, opt->lb, opt->ub) == opt->n)
        return 0;
    switch (opt->algorithm) {
    case NLOPT_GN_DIRECT:
    case NLOPT_GN_DIRECT_L:
    case NLOPT_GN_DIRECT_L_RAND:
    case NLOPT_GN_DIRECT_NOSCAL:
    case NLOPT_GN_DIRECT_L_NOSCAL:
    case NLOPT_GN_DIRECT_L_RAND_NOSCAL:
    case NLOPT_GN_ORIG_DIRECT:
    case NLOPT_GN_ORIG_DIRECT_L:
    case NLOPT_GN_CRS2_LM:
    case NLOPT_LN_COBYLA:
    case NLOPT_LN_NELDERMEAD:
    case NLOPT_LN_SBPLX:
    case NLOPT_GN_ISRES:
    case NLOPT_GD_STOGO:
    case NLOPT_GD_STOGO_RAND:
        return 1;

    default:
        return 0;
    }
}

/*********************************************************************/

#define POP(defaultpop) (opt->stochastic_population > 0 ? opt->stochastic_population : (nlopt_stochastic_population > 0 ? nlopt_stochastic_population : (defaultpop)))

/* unlike nlopt_optimize() below, only handles minimization case */
static nlopt_result nlopt_optimize_(nlopt_opt opt, double *x, double *minf)
{
    const double *lb, *ub;
    nlopt_algorithm algorithm;
    nlopt_func f;
    void *f_data;
    unsigned n, i;
    int ni;
    nlopt_stopping stop;

    if (!opt || !x || !minf || !opt->f || opt->maximize)
        RETURN_ERR(NLOPT_INVALID_ARGS, opt, "NULL args to nlopt_optimize_");

    /* reset stopping flag */
    nlopt_set_force_stop(opt, 0);
    opt->force_stop_child = NULL;

    /* copy a few params to local vars for convenience */
    n = opt->n;
    ni = (int) n;               /* most of the subroutines take "int" arg */
    lb = opt->lb;
    ub = opt->ub;
    algorithm = opt->algorithm;
    f = opt->f;
    f_data = opt->f_data;

    if (n == 0) {               /* trivial case: no degrees of freedom */
        *minf = opt->f(n, x, NULL, opt->f_data);
        return NLOPT_SUCCESS;
    }

    *minf = HUGE_VAL;

    /* make sure rand generator is inited */
    nlopt_srand_time_default(); /* default is non-deterministic */

    /* check bound constraints */
    for (i = 0; i < n; ++i)
        if (lb[i] > ub[i] || x[i] < lb[i] || x[i] > ub[i]) {
            nlopt_set_errmsg(opt, "bounds %d fail %g <= %g <= %g", i, lb[i], x[i], ub[i]);
            return NLOPT_INVALID_ARGS;
        }

    stop.n = n;
    stop.minf_max = opt->stopval;
    stop.ftol_rel = opt->ftol_rel;
    stop.ftol_abs = opt->ftol_abs;
    stop.xtol_rel = opt->xtol_rel;
    stop.xtol_abs = opt->xtol_abs;
    stop.x_weights = opt->x_weights;
    opt->numevals = 0;
    stop.nevals_p = &(opt->numevals);
    stop.maxeval = opt->maxeval;
    stop.maxtime = opt->maxtime;
    stop.start = nlopt_seconds();
    stop.force_stop = &(opt->force_stop);
    stop.stop_msg = &(opt->errmsg);

    switch (algorithm) {

#if 0
        /* lacking a free/open-source license, we no longer use
           Rowan's code, and instead use by "sbplx" re-implementation */
    case NLOPT_LN_SUBPLEX:
        {
            int iret, freedx = 0;
            if (!opt->dx) {
                freedx = 1;
                if (nlopt_set_default_initial_step(opt, x) != NLOPT_SUCCESS)
                    return NLOPT_OUT_OF_MEMORY;
            }
            iret = nlopt_subplex(f_bound, minf, x, n, opt, &stop, opt->dx);
            if (freedx) {
                free(opt->dx);
                opt->dx = NULL;
            }
            switch (iret) {
            case -2:
                return NLOPT_INVALID_ARGS;
            case -20:
                return NLOPT_FORCED_STOP;
            case -10:
                return NLOPT_MAXTIME_REACHED;
            case -1:
                return NLOPT_MAXEVAL_REACHED;
            case 0:
                return NLOPT_XTOL_REACHED;
            case 1:
                return NLOPT_SUCCESS;
            case 2:
                return NLOPT_MINF_MAX_REACHED;
            case 20:
                return NLOPT_FTOL_REACHED;
            case -200:
                return NLOPT_OUT_OF_MEMORY;
            default:
                return NLOPT_FAILURE;   /* unknown return code */
            }
            break;
        }
#endif

    case NLOPT_LD_VAR1:

    case NLOPT_LD_TNEWTON:
    case NLOPT_LD_TNEWTON_RESTART:
    case NLOPT_LD_TNEWTON_PRECOND:


    case NLOPT_LD_MMA:
    case NLOPT_LD_CCSAQ:
        {
            nlopt_opt dual_opt;
            nlopt_result ret;
#define LO(param, def) (opt->local_opt ? opt->local_opt->param : (def))
            dual_opt = nlopt_create((nlopt_algorithm)nlopt_get_param(opt, "dual_algorithm", LO(algorithm, nlopt_local_search_alg_deriv)),
                                    nlopt_count_constraints(opt->m, opt->fc));
            if (!dual_opt)
                RETURN_ERR(NLOPT_FAILURE, opt, "failed creating dual optimizer");
            nlopt_set_ftol_rel(dual_opt, nlopt_get_param(opt, "dual_ftol_rel", LO(ftol_rel, 1e-14)));
            nlopt_set_ftol_abs(dual_opt, nlopt_get_param(opt, "dual_ftol_abs", LO(ftol_abs, 0.0)));
            nlopt_set_xtol_rel(dual_opt, nlopt_get_param(opt, "dual_xtol_rel", 0.0));
            nlopt_set_xtol_abs1(dual_opt, nlopt_get_param(opt, "dual_xtol_abs", 0.0));
            nlopt_set_maxeval(dual_opt, nlopt_get_param(opt, "dual_maxeval", LO(maxeval, 100000)));
#undef LO

            if (algorithm == NLOPT_LD_MMA)
                ret = mma_minimize(n, f, f_data, opt->m, opt->fc, lb, ub, x, minf, &stop, dual_opt, (int)nlopt_get_param(opt, "inner_maxeval",0), (unsigned)nlopt_get_param(opt, "verbosity",0));
            else
                ret = ccsa_quadratic_minimize(n, f, f_data, opt->m, opt->fc, opt->pre, lb, ub, x, minf, &stop, dual_opt, (int)nlopt_get_param(opt, "inner_maxeval",0), (unsigned)nlopt_get_param(opt, "verbosity",0));
            nlopt_destroy(dual_opt);
            return ret;
        }

    case NLOPT_LN_COBYLA:
        {
            nlopt_result ret;
            int freedx = 0;
            if (!opt->dx) {
                freedx = 1;
                if (nlopt_set_default_initial_step(opt, x) != NLOPT_SUCCESS)
                    RETURN_ERR(NLOPT_OUT_OF_MEMORY, opt, "failed to allocate initial step");
            }
            ret = cobyla_minimize(n, f, f_data, opt->m, opt->fc, opt->p, opt->h, lb, ub, x, minf, &stop, opt->dx);
            if (freedx) {
                free(opt->dx);
                opt->dx = NULL;
            }
            return ret;
        }

    case NLOPT_LN_NELDERMEAD:
    case NLOPT_LN_SBPLX:
        {
            nlopt_result ret;
            int freedx = 0;
            if (!opt->dx) {
                freedx = 1;
                if (nlopt_set_default_initial_step(opt, x) != NLOPT_SUCCESS)
                    RETURN_ERR(NLOPT_OUT_OF_MEMORY, opt, "failed to allocate initial step");
            }
            if (algorithm == NLOPT_LN_NELDERMEAD)
                ret = nldrmd_minimize(ni, f, f_data, lb, ub, x, minf, opt->dx, &stop);
            else
                ret = sbplx_minimize(ni, f, f_data, lb, ub, x, minf, opt->dx, &stop);
            if (freedx) {
                free(opt->dx);
                opt->dx = NULL;
            }
            return ret;
        }
    

    default:
        return NLOPT_INVALID_ARGS;
    }

    return NLOPT_SUCCESS;       /* never reached */
}

/*********************************************************************/

typedef struct {
    nlopt_func f;
    nlopt_precond pre;
    void *f_data;
} f_max_data;

/* wrapper for maximizing: just flip the sign of f and grad */
static double f_max(unsigned n, const double *x, double *grad, void *data)
{
    f_max_data *d = (f_max_data *) data;
    double val = d->f(n, x, grad, d->f_data);
    if (grad) {
        unsigned i;
        for (i = 0; i < n; ++i)
            grad[i] = -grad[i];
    }
    return -val;
}

static void pre_max(unsigned n, const double *x, const double *v, double *vpre, void *data)
{
    f_max_data *d = (f_max_data *) data;
    unsigned i;
    d->pre(n, x, v, vpre, d->f_data);
    for (i = 0; i < n; ++i)
        vpre[i] = -vpre[i];
}

nlopt_result NLOPT_STDCALL nlopt_optimize(nlopt_opt opt, double *x, double *opt_f)
{
    nlopt_func f;
    void *f_data;
    nlopt_precond pre;
    f_max_data fmd;
    int maximize;
    nlopt_result ret;

    nlopt_unset_errmsg(opt);
    if (!opt || !opt_f || !opt->f)
        RETURN_ERR(NLOPT_INVALID_ARGS, opt, "NULL args to nlopt_optimize");
    f = opt->f;
    f_data = opt->f_data;
    pre = opt->pre;

    /* for maximizing, just minimize the f_max wrapper, which
       flips the sign of everything */
    if ((maximize = opt->maximize)) {
        fmd.f = f;
        fmd.f_data = f_data;
        fmd.pre = pre;
        opt->f = f_max;
        opt->f_data = &fmd;
        if (opt->pre)
            opt->pre = pre_max;
        opt->stopval = -opt->stopval;
        opt->maximize = 0;
    }

    {                           /* possibly eliminate lb == ub dimensions for some algorithms */
        nlopt_opt elim_opt = opt;
        if (elimdim_wrapcheck(opt)) {
            elim_opt = elimdim_create(opt);
            if (!elim_opt) {
                nlopt_set_errmsg(opt, "failure allocating elim_opt");
                ret = NLOPT_OUT_OF_MEMORY;
                goto done;
            }
            elimdim_shrink(opt->n, x, opt->lb, opt->ub);
            opt->force_stop_child = elim_opt;
        }

        ret = nlopt_optimize_(elim_opt, x, opt_f);

        if (elim_opt != opt) {
            opt->numevals = elim_opt->numevals;
            opt->errmsg = elim_opt->errmsg; elim_opt->errmsg = NULL;
            elimdim_destroy(elim_opt);
            elimdim_expand(opt->n, x, opt->lb, opt->ub);
            opt->force_stop_child = NULL;
        }
    }

  done:
    if (maximize) {             /* restore original signs */
        opt->maximize = maximize;
        opt->stopval = -opt->stopval;
        opt->f = f;
        opt->f_data = f_data;
        opt->pre = pre;
        *opt_f = -*opt_f;
    }

    return ret;
}

/*********************************************************************/

nlopt_result nlopt_optimize_limited(nlopt_opt opt, double *x, double *minf, int maxeval, double maxtime)
{
    int save_maxeval;
    double save_maxtime;
    nlopt_result ret;

    nlopt_unset_errmsg(opt);

    if (!opt)
        RETURN_ERR(NLOPT_INVALID_ARGS, opt, "NULL opt arg");

    save_maxeval = nlopt_get_maxeval(opt);
    save_maxtime = nlopt_get_maxtime(opt);

    /* override opt limits if maxeval and/or maxtime are more stringent */
    if (save_maxeval <= 0 || (maxeval > 0 && maxeval < save_maxeval))
        nlopt_set_maxeval(opt, maxeval);
    if (save_maxtime <= 0 || (maxtime > 0 && maxtime < save_maxtime))
        nlopt_set_maxtime(opt, maxtime);

    ret = nlopt_optimize(opt, x, minf);

    nlopt_set_maxeval(opt, save_maxeval);
    nlopt_set_maxtime(opt, save_maxtime);

    return ret;
}

/*********************************************************************/
